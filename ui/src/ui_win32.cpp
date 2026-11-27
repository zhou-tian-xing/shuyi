// 数弈 — native Win32/GDI user interface.  No web view, browser, or HTML.
//
// The window is one board and three pieces of chrome:
//
//   menu         文件 / 模式 / 规则 / 引擎 / 帮助 — everything that is not a move;
//   left         what the analysis produced: the verdict, the move list, the
//                score, and the curves of the line on screen;
//   right        the record tree: every move ever played from this game, with a
//                branch per alternative, so going back is navigation, not undo;
//   bottom       the A value and the pieces still in hand, small.
//
// Everything else lives in the menu bar (four items) or in the help window.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shellapi.h>

#include "discovery.h"
#include "engine_client.h"
#include "engine_view.h"
#include "game_record.h"
#include "shuyi/format.h"
#include "shuyi/io.h"
#include "shuyi/position.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace shuyi;

namespace {

// ---------------------------------------------------------------------------
// Command ids
// ---------------------------------------------------------------------------
constexpr int ID_NEW_GAME = 1001;
constexpr int ID_SAVE_RECORD = 1002;
constexpr int ID_LOAD_RECORD = 1003;
constexpr int ID_EXIT = 1004;

constexpr int ID_MODE_PVP = 2001;
constexpr int ID_MODE_HUMAN_RED = 2002;
constexpr int ID_MODE_HUMAN_BLACK = 2003;

constexpr int ID_A_PLUS = 3001;
constexpr int ID_A_MINUS = 3002;

constexpr int ID_GUIDE = 4001;
constexpr int ID_PROJECT_PAGE = 4002;
constexpr int ID_ABOUT = 4003;

constexpr int ID_RULESET_CLASSIC = 5101;
constexpr int ID_RULESET_BLOCKING = 5102;
constexpr int ID_RULESET_NOPASS = 5103;
constexpr int ID_ENGINE_OFF = 5001;
constexpr int ID_ENGINE_MCTS = 5003;
constexpr int ID_ENGINE_MCTS_NN = 5004;  // popup: one entry per model file
constexpr int ID_ENGINE_PICK = 5007;
constexpr int ID_ENGINE_AUTO = 5008;
constexpr int ID_ENGINE_INFO = 5009;
constexpr int ID_SETTINGS = 5010;
constexpr int ID_ENGINE_MOVE = 5011;
constexpr int ID_MODEL_RESCAN = 6090;
constexpr int ID_MODEL_BASE = 6100;  // + model index
constexpr int ID_MODEL_LIMIT = 6400; // exclusive upper bound, below nothing else
constexpr int ID_SETTINGS_OK = 6001;
constexpr int ID_SETTINGS_CANCEL = 6002;

constexpr const wchar_t* kProjectUrl = L"https://github.com/zhou-tian-xing/shuyi";

// Simulations per analysis and per move are settings (default 512).  A visit
// budget is the only machine-independent unit: 512 visits is 512 visits on the
// laptop and on the GPU, so two runs of the same position are comparable.
constexpr int kDefaultVisits = 512;
constexpr const wchar_t* kSettingsClass = L"ShuyiSettingsWindow";

constexpr UINT WM_APP_RESET = WM_APP + 1;

enum class Mode { Pvp, HumanRed, HumanBlack };
enum class EngineChoice { Off, Mcts, MctsNn };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
HWND g_hwnd = nullptr;
HMENU g_menu = nullptr;
HMENU g_modeMenu = nullptr;
HMENU g_rulesetMenu = nullptr;
HMENU g_engineMenu = nullptr;
HMENU g_modelMenu = nullptr;
int g_modelMenuPosition = -1;

struct ModelEntry {
    // Win32 *A functions speak the ANSI code page, so paths found by scanning are
    // ANSI; the engine re-parses its command line as UTF-8.  Keep both.
    std::string path;        // ANSI, for Win32 / fopen
    std::string enginePath;  // UTF-8, what set_model receives
    std::wstring label;      // "run-name/file.bin", unique in the menu
    int ruleset = 0;         // 0 classic, 1 blocking, -1 unknown
    int steps = 0;           // training steps from "-s<steps>-", 0 when unknown
};
std::vector<ModelEntry> g_modelList;
std::string g_nnModelPath;
// The engine is a separate program.  Its path is a user setting (shuyi-ui.ini).
std::wstring g_enginePath;
std::wstring g_enginePathUsed;
bool g_engineAuto = true;
HINSTANCE g_instance = nullptr;

// The record owns the game; the position is always "the node on screen".
GameRecord g_record;
Position g_position;
// Which record node the running analysis belongs to (-1: none).  The analysis is
// cached *on that node*, so navigating away and back shows the same numbers.
int g_analysisNode = -1;
// Which analysis run `g_analysisNode` belongs to.  One run is one search tree:
// it starts when the tree does and lives until the engine's board is rebuilt
// (`clear_board`, a new position, a new process).  Visits are only comparable
// inside a run, so nothing may be refused across two of them.
int g_analysisRun = 0;

constexpr double kDefaultAValue = 0.5;
constexpr double kAValueStep = 0.5;
double g_aValue = kDefaultAValue;
int g_selected = 1;
int g_hoverCell = -1;
bool g_thinking = false;
Mode g_mode = Mode::Pvp;
int g_dpi = 96;
int g_treeScroll = 0;

EngineClient g_engineClient;
EngineChoice g_engineChoice = EngineChoice::Off;
bool g_analysisOn = false;
bool g_analysisFinished = false;
int g_analysisVisits = kDefaultVisits;
int g_playVisits = kDefaultVisits;
bool g_enginePending = false;
std::string g_pendingReplyId;
int g_commandId = 0;
std::wstring g_engineMessage = L"引擎未启用";
bool g_engineReady = false;
Ruleset g_ruleset = kDefaultRuleset;
// A record given on the command line is loaded once the window exists.  Loading it
// before the message loop runs would be undone by the queued reset.
std::wstring g_pendingOpen;

HFONT g_fontTitle = nullptr;
HFONT g_fontSubtitle = nullptr;
HFONT g_fontTurn = nullptr;
HFONT g_fontBig = nullptr;
HFONT g_fontPiece = nullptr;
HFONT g_fontUi = nullptr;
HFONT g_fontSmall = nullptr;
HFONT g_fontCoord = nullptr;
HFONT g_fontTiny = nullptr;  // the superscript piece count

const COLORREF kBackground = RGB(22, 26, 33);
const COLORREF kPanel = RGB(36, 41, 50);
const COLORREF kPanelSoft = RGB(45, 51, 62);
const COLORREF kPanelDeep = RGB(29, 33, 41);
const COLORREF kBoard = RGB(225, 201, 153);
const COLORREF kBoardLine = RGB(105, 77, 44);
const COLORREF kText = RGB(238, 233, 224);
const COLORREF kMuted = RGB(166, 170, 178);
const COLORREF kGold = RGB(229, 184, 94);
const COLORREF kRed = RGB(191, 52, 52);
const COLORREF kRedSoft = RGB(224, 139, 125);
const COLORREF kBlack = RGB(32, 35, 42);
const COLORREF kBlackSoft = RGB(128, 133, 144);
const COLORREF kGreen = RGB(105, 188, 133);
const COLORREF kCyan = RGB(100, 198, 244);

int queryDpi() {
    HDC screen = GetDC(nullptr);
    int dpi = screen ? GetDeviceCaps(screen, LOGPIXELSY) : 96;
    if (screen) ReleaseDC(nullptr, screen);
    return dpi > 0 ? dpi : 96;
}

int scale(int value) { return MulDiv(value, g_dpi, 96); }

COLORREF mixColor(COLORREF a, COLORREF b, double t) {
    t = std::clamp(t, 0.0, 1.0);
    const auto mix = [&](int x, int y) {
        return static_cast<int>(std::lround(x * (1.0 - t) + y * t));
    };
    return RGB(mix(GetRValue(a), GetRValue(b)), mix(GetGValue(a), GetGValue(b)),
               mix(GetBValue(a), GetBValue(b)));
}

void refresh(bool immediate = false);
void maybeAiMove();
void setAValue(double value);
void syncEngineChecks();
void syncEnginePosition();
void openSettings();
void goToNode(int id);
int treeScrollFor(int id);
void sendEngineStopIfRunning();
bool sendEngine(const std::string& line);
void startEngineAnalysis();
bool loadRecordFromPath(const std::wstring& path);
bool loadRecordFromText(const std::string& text);

// ---------------------------------------------------------------------------
// Small GDI helpers
// ---------------------------------------------------------------------------
void fillRectColor(HDC dc, const RECT& rect, COLORREF color) {
    HBRUSH brush = CreateSolidBrush(color);
    FillRect(dc, &rect, brush);
    DeleteObject(brush);
}

void roundRect(HDC dc, const RECT& rect, int radius, COLORREF fill, COLORREF border = CLR_INVALID,
               int borderWidth = 1) {
    HBRUSH brush = CreateSolidBrush(fill);
    HPEN pen = border == CLR_INVALID ? CreatePen(PS_NULL, 0, fill)
                                     : CreatePen(PS_SOLID, borderWidth, border);
    HGDIOBJ oldBrush = SelectObject(dc, brush);
    HGDIOBJ oldPen = SelectObject(dc, pen);
    RoundRect(dc, rect.left, rect.top, rect.right, rect.bottom, radius, radius);
    SelectObject(dc, oldPen);
    SelectObject(dc, oldBrush);
    DeleteObject(pen);
    DeleteObject(brush);
}

void line(HDC dc, int x1, int y1, int x2, int y2, COLORREF color, int width = 1) {
    HPEN pen = CreatePen(PS_SOLID, width, color);
    HGDIOBJ old = SelectObject(dc, pen);
    MoveToEx(dc, x1, y1, nullptr);
    LineTo(dc, x2, y2);
    SelectObject(dc, old);
    DeleteObject(pen);
}

void drawTextRect(HDC dc, const std::wstring& text, RECT rect, HFONT font, COLORREF color,
                  UINT flags = DT_CENTER | DT_VCENTER | DT_SINGLELINE) {
    HGDIOBJ oldFont = SelectObject(dc, font);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, color);
    DrawTextW(dc, text.c_str(), -1, &rect, flags | DT_NOPREFIX);
    SelectObject(dc, oldFont);
}

void drawTextAt(HDC dc, const std::wstring& text, int x, int y, int width, int height, HFONT font,
                COLORREF color, UINT flags = DT_CENTER | DT_VCENTER | DT_SINGLELINE) {
    RECT rect{x, y, x + width, y + height};
    drawTextRect(dc, text, rect, font, color, flags);
}

void filledCircle(HDC dc, int cx, int cy, int radius, COLORREF fill, COLORREF border = CLR_INVALID,
                  int borderWidth = 1) {
    HBRUSH brush = CreateSolidBrush(fill);
    HPEN pen = border == CLR_INVALID ? CreatePen(PS_NULL, 0, fill)
                                     : CreatePen(PS_SOLID, borderWidth, border);
    HGDIOBJ oldBrush = SelectObject(dc, brush);
    HGDIOBJ oldPen = SelectObject(dc, pen);
    Ellipse(dc, cx - radius, cy - radius, cx + radius, cy + radius);
    SelectObject(dc, oldPen);
    SelectObject(dc, oldBrush);
    DeleteObject(pen);
    DeleteObject(brush);
}

std::wstring signedText(int value) {
    if (value > 0) return L"+" + std::to_wstring(value);
    return std::to_wstring(value);
}

// The A value lives on a half-integer grid, so it prints without trailing zeros.
std::string formatAValue(double value) {
    char buffer[32]{};
    std::snprintf(buffer, sizeof(buffer), "%.2f", value);
    std::string text(buffer);
    while (text.size() > 1 && text.back() == '0') text.pop_back();
    if (!text.empty() && text.back() == '.') text.pop_back();
    return text;
}

std::wstring formatAValueW(double value) {
    const std::string text = formatAValue(value);
    return std::wstring(text.begin(), text.end());
}

std::wstring signedValue(double value) {
    return value > 0.0 ? L"+" + formatAValueW(value) : formatAValueW(value);
}

std::wstring signedDecimal(double value) {
    if (value == 0.0) value = 0.0;  // never "+-0.0"
    std::wostringstream out;
    if (std::fabs(value) < 0.05) value = 0.0;
    if (value > 0.0) out << L'+';
    out << std::fixed << std::setprecision(1) << value;
    return out.str();
}

double normalizeAValue(double value) {
    const double bounded = std::clamp(value, -40.0, 40.0);
    const double rounded = std::round(bounded / kAValueStep) * kAValueStep;
    return rounded == 0.0 ? 0.0 : rounded;
}

const wchar_t* outcomeText(Outcome outcome) {
    switch (outcome) {
    case Outcome::RedWin: return L"红胜";
    case Outcome::BlackWin: return L"黑胜";
    case Outcome::Draw: return L"平局";
    default: return L"进行中";
    }
}

bool humanToMove() {
    if (g_position.gameOver()) return false;
    if (g_mode == Mode::Pvp) return true;
    if (g_mode == Mode::HumanRed) return g_position.sideToMove() == Player::Red;
    return g_position.sideToMove() == Player::Black;
}

bool aiToMove() {
    return !g_position.gameOver() && !humanToMove();
}

int firstAvailable(Player player) {
    for (int n = 1; n <= 9; ++n) {
        if (g_position.remaining(player, n) > 0) return n;
    }
    return 1;
}

std::wstring widen(const std::string& text) {
    std::wstring result;
    result.reserve(text.size());
    for (unsigned char c : text) result.push_back(static_cast<wchar_t>(c));
    return result;
}

// ---------------------------------------------------------------------------
// The record and the position on screen
// ---------------------------------------------------------------------------
EngineAnalysisView& currentAnalysis() {
    return g_record.node(g_record.current()).analysis;
}

// Rebuild the position by replaying the tree from the root.  A record is the
// source of truth: the position is derived, never edited in place, so what the
// board shows and what the tree says can never disagree.
void rebuildPosition() {
    g_position.reset();
    g_position.setRuleset(g_ruleset);
    for (int id : g_record.line(g_record.current())) {
        const Move move = g_record.node(id).move;
        if (isPass(move)) {
            g_position.pass();
        } else if (!g_position.play(move)) {
            break;  // a corrupt record must not spin
        }
    }
    if (humanToMove()) g_selected = firstAvailable(g_position.sideToMove());
}

void clearAnalysisState() {
    g_analysisOn = false;
    g_analysisFinished = false;
    g_analysisNode = -1;
}

// The window is a review surface: a position where the computer is to move does
// not play itself.  Navigating back one move used to make the engine answer
// immediately, which silently appended a branch to the record -- after a detour
// through that branch every later move looked like a variation.  Say what the
// state is instead, and leave the moving to the menu.
constexpr const wchar_t* kComputerToMove = L"轮到电脑：引擎菜单 → 让电脑走一步";
void noteComputerToMove() {
    if (aiToMove()) {
        g_engineMessage = kComputerToMove;
    } else if (g_engineMessage == kComputerToMove) {
        // Leaving that position must take the note with it: a stale "the computer
        // is to move" on a position where it is not would be a lie in the panel.
        g_engineMessage = g_engineChoice == EngineChoice::Off ? L"引擎未启用" : L"引擎已同步";
    }
}

// The move in the engine's own spelling ("a4 2"), the form `play` takes.
std::string playCommand(Player side, Move move) {
    std::string line = std::string("play ") + (side == Player::Red ? "red" : "black") + " ";
    if (isPass(move)) return line + "pass";
    return line + squareName(move.cell) + " " + std::to_string(static_cast<int>(move.number));
}

// Navigation: the same record, another node.  The panel shows that node's cached
// analysis instead of whatever happened to be on screen before.
//
// Stepping forward is a *move* for the engine, not a new position: `play` lets it
// carry the subtree it just searched into the child, so an analysis started there
// continues from the visits the parent's search already spent on it.  Going up (or
// jumping across the tree) has to rebuild the board from the root, which throws
// the tree away -- the numbers then start again and the panel says so.
void goToNode(int id) {
    if (!g_record.contains(id)) return;
    if (id == g_record.current()) {
        refresh();
        return;
    }
    const int from = g_record.current();
    sendEngineStopIfRunning();
    // The move is spelled before the record moves: the side to move at the child
    // is the opponent's, so depth is what says whose move it was.
    const int ply = g_record.depth(id) - 1;
    const Player mover = (ply % 2 == 0) ? Player::Red : Player::Black;
    const Move move = g_record.node(id).move;
    const bool forward = g_engineClient.running() && g_record.parentOf(id) == from;
    clearAnalysisState();
    g_record.select(id);
    rebuildPosition();
    g_enginePending = false;
    g_pendingReplyId.clear();
    g_thinking = false;
    if (g_engineChoice != EngineChoice::Off && g_engineReady) {
        if (forward) {
            sendEngine(playCommand(mover, move));
            ++g_analysisRun;
        } else {
            syncEnginePosition();
        }
    }
    noteComputerToMove();
    refresh(true);
}

void sendEngineStopIfRunning() {
    if (g_engineClient.running()) g_engineClient.send("stop");
}

// Plays `move` from the node on screen.  The record is updated first, then the
// position is rebuilt from it, then the engine is told -- one order everywhere,
// so the three can never drift apart.
void recordMove(Move move, Player mover) {
    // Analysis is a mode, not a one-off: it continues on the new position unless
    // the user had stopped it.  A run that reached the visit cap is not a stop --
    // the new position has its own budget, and has not spent a visit of it.
    const bool wasAnalyzing = g_analysisOn || g_analysisFinished;
    const int from = g_record.current();
    int id = g_record.play(from, move);
    // `play` may carry the subtree of this move over into the new position, in
    // which case the analysis continues where the parent's search left it.  It may
    // also rebuild the tree.  Either way the numbers from here on belong to a new
    // run, and are compared with that run's own figures, not with the old ones.
    ++g_analysisRun;
    if (g_engineClient.running()) {
        g_engineClient.send("stop");
        g_enginePending = false;
        g_analysisFinished = false;
        g_engineClient.send("play " + std::string(mover == Player::Red ? "red" : "black") + " " +
                            (isPass(move) ? std::string("pass") : squareName(move.cell) + " " +
                                                                    std::to_string(static_cast<int>(move.number))));
    }
    rebuildPosition();
    // A forced pass is a real move in this game (blocking rule): without a node
    // for it the record could not reproduce the position.
    int guard = 0;
    while (g_position.mustPass() && guard++ < 2) {
        const Player side = g_position.sideToMove();
        id = g_record.play(id, Move(kPassCell, kPassNumber));
        if (g_engineClient.running()) {
            g_engineClient.send(std::string("play ") + (side == Player::Red ? "red" : "black") +
                                " pass");
        }
        rebuildPosition();
    }
    clearAnalysisState();
    g_treeScroll = treeScrollFor(g_record.current());
    refresh(true);
    if (wasAnalyzing && !g_position.gameOver() && !g_enginePending) startEngineAnalysis();
}

// ---------------------------------------------------------------------------
// External engine client
// ---------------------------------------------------------------------------
const char* engineChoiceName(EngineChoice choice) {
    switch (choice) {
    case EngineChoice::Mcts: return "mcts";
    case EngineChoice::MctsNn: return "mcts_nn";
    default: return "off";
    }
}

std::wstring fileNameOf(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    return widen(name);
}

// What the search has spent on this position, and how fast it is spending it.
// The model is the user's own choice, so naming it in the panel told them nothing
// they had not just selected; the effort is the number that changes while they
// watch, and the one that explains why an estimate is what it is.
std::wstring analysisEffortText(const EngineAnalysisView& view) {
    // A solved position was not sampled: it has an answer and a node count, and a
    // rate would be a number about the clock, not about the search.
    if (view.proven && view.visits == 0) {
        return L"精确解算 · " + formatVisits(view.nodes) + L" 节点";
    }
    std::wstring text = formatVisits(view.visits) + L" 次";
    if (view.elapsedMs > 0) {
        const double rate =
            static_cast<double>(view.visits) * 1000.0 / static_cast<double>(view.elapsedMs);
        text += L" · " + formatVisits(static_cast<long long>(std::llround(rate))) + L"/秒";
    }
    if (view.proven) text += L" · 已解算";
    if (view.backend == "cuda") text += L" · GPU";
    return text;
}

bool sendEngine(const std::string& line) {
    if (!g_engineClient.running()) return false;
    return g_engineClient.send(line);
}

void applyEngineAnalysis(int node, const std::string& payload) {
    EngineAnalysisView view;
    std::string error;
    if (!parseAnalysis(payload, view, &error)) {
        g_engineMessage = L"引擎分析数据无法解析：" + widen(error);
        return;
    }
    g_engineMessage = g_enginePending ? L"引擎思考中…" : L"分析中…";
    if (!g_record.contains(node)) return;
    RecordNode& target = g_record.node(node);
    if (!analysisReplaces(view, g_analysisRun, target.analysis, target.run)) return;
    g_record.setAnalysis(node, std::move(view), g_analysisRun);
}

void startEngineAnalysis(int visits) {
    if (g_engineChoice == EngineChoice::Off || !g_engineClient.running() || g_position.gameOver()) return;
    g_enginePending = false;
    sendEngine("stop");
    g_analysisOn = true;
    g_analysisFinished = false;
    g_analysisNode = g_record.current();
    sendEngine("analyze " + std::string(g_position.sideToMove() == Player::Red ? "red" : "black") +
               " " + std::to_string(std::max(0, visits)));
    g_engineMessage = visits > 0 ? L"分析中" : L"分析中（不限访问）";
    refresh();
}

void startEngineAnalysis() { startEngineAnalysis(g_analysisVisits); }

void stopEngineAnalysis() {
    g_analysisOn = false;
    sendEngineStopIfRunning();
    g_engineMessage = g_analysisFinished ? L"已达访问上限" : L"已暂停";
    refresh();
}

// Space: start, stop, or continue past the limit.  Stopping keeps the numbers on
// screen -- they are still the best information about this position.
void toggleAnalysis() {
    if (g_position.gameOver() || g_engineChoice == EngineChoice::Off) return;
    // Space must not cancel a move the computer is in the middle of choosing.
    if (g_enginePending) return;
    if (g_analysisOn) {
        stopEngineAnalysis();
        return;
    }
    if (!g_engineReady) return;
    // Continuing after a finished run ignores the budget: the visit limit is a
    // stopping point, not a cap on what may be searched.
    startEngineAnalysis(g_analysisFinished ? 0 : g_analysisVisits);
}

void syncEnginePosition() {
    if (!g_engineClient.running()) return;
    g_enginePending = false;
    g_analysisFinished = false;
    sendEngine("stop");
    sendEngine("clear_board");
    // The engine forgets its tree here, so every number it sends from now on
    // belongs to a new run.
    ++g_analysisRun;
    for (int id : g_record.line(g_record.current())) {
        const Move move = g_record.node(id).move;
        // The side alternates from Red at the root; a pass is a move too.
        const int ply = g_record.depth(id) - 1;
        const char* side = (ply % 2 == 0) ? "red" : "black";
        if (isPass(move)) {
            sendEngine(std::string("play ") + side + " pass");
        } else {
            sendEngine(std::string("play ") + side + " " + squareName(move.cell) + " " +
                       std::to_string(static_cast<int>(move.number)));
        }
    }
    g_engineMessage = L"引擎已同步";
    if (g_analysisOn && !g_position.gameOver()) startEngineAnalysis();
}

void stopEngineProcess() {
    g_engineClient.stop();
    g_engineReady = false;
    g_analysisOn = false;
    g_enginePending = false;
    g_analysisNode = -1;
    g_engineMessage = L"引擎未启用";
}

// ANSI (what the *A APIs return) <-> UTF-8 (what argv and the engine use).
std::string ansiToUtf8(const std::string& text) {
    if (text.empty()) return text;
    const int wideLength = MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, nullptr, 0);
    if (wideLength <= 1) return text;
    std::wstring wide(static_cast<std::size_t>(wideLength - 1), L'\0');
    MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, wide.data(), wideLength);
    const int utf8Length =
        WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (utf8Length <= 1) return text;
    std::string out(static_cast<std::size_t>(utf8Length - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, out.data(), utf8Length, nullptr, nullptr);
    return out;
}

std::wstring widenUtf8(const std::string& text) {
    if (text.empty()) return std::wstring();
    const int length = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (length <= 1) return std::wstring();
    std::wstring wide(static_cast<std::size_t>(length - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), length);
    return wide;
}

std::wstring widenAnsi(const std::string& text) {
    if (text.empty()) return std::wstring();
    const int wideLength = MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, nullptr, 0);
    if (wideLength <= 1) return std::wstring();
    std::wstring wide(static_cast<std::size_t>(wideLength - 1), L'\0');
    MultiByteToWideChar(CP_ACP, 0, text.c_str(), -1, wide.data(), wideLength);
    return wide;
}

std::string narrowUtf8(const std::wstring& text) {
    if (text.empty()) return std::string();
    const int length = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (length <= 1) return std::string();
    std::string out(static_cast<std::size_t>(length - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, out.data(), length, nullptr, nullptr);
    return out;
}

std::string trimAscii(const std::string& text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && (text[begin] == ' ' || text[begin] == '\t')) ++begin;
    while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t')) --end;
    return text.substr(begin, end - begin);
}

// ---------------------------------------------------------------------------
// Model discovery
//
// The window can be started from anywhere, so "models/" relative to the current
// directory is not enough.  Scan a few roots and inside each collect every *.bin
// that lives in a directory called "models".
// ---------------------------------------------------------------------------
std::string exeDirectory() {
    char buffer[MAX_PATH]{};
    const DWORD length = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) return std::string();
    std::string path(buffer, length);
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

std::string normalizedPath(const std::string& path) {
    char buffer[MAX_PATH]{};
    const DWORD length = GetFullPathNameA(path.c_str(), MAX_PATH, buffer, nullptr);
    if (length == 0 || length >= MAX_PATH) return path;
    std::string result(buffer, length);
    while (!result.empty() && (result.back() == '\\' || result.back() == '/')) result.pop_back();
    return result;
}

bool skipDirectory(const char* name);  // defined below

std::vector<std::string> modelRoots() {
    std::vector<std::string> roots;
    const auto add = [&](const std::string& candidate) {
        if (candidate.empty()) return;
        const std::string full = normalizedPath(candidate);
        if (GetFileAttributesA(full.c_str()) == INVALID_FILE_ATTRIBUTES) return;
        const std::size_t slash = full.find_last_of("\\/");
        const char* base = slash == std::string::npos ? full.c_str() : full.c_str() + slash + 1;
        if (skipDirectory(base)) return;
        for (const std::string& seen : roots) {
            if (_stricmp(seen.c_str(), full.c_str()) == 0) return;
        }
        roots.push_back(full);
    };
    const std::string exe = exeDirectory();
    add(exe);
    add(exe + "\\..");
    char cwd[MAX_PATH]{};
    if (GetCurrentDirectoryA(MAX_PATH, cwd) > 0) add(cwd);
    char env[MAX_PATH * 4]{};
    const DWORD envLength = GetEnvironmentVariableA("SHUYI_MODEL_DIRS", env, sizeof(env));
    if (envLength > 0 && envLength < sizeof(env)) {
        std::string list(env, envLength);
        std::size_t begin = 0;
        while (begin <= list.size()) {
            const std::size_t sep = list.find(';', begin);
            const std::string item =
                list.substr(begin, sep == std::string::npos ? std::string::npos : sep - begin);
            if (!item.empty()) add(item);
            if (sep == std::string::npos) break;
            begin = sep + 1;
        }
    }
    return roots;
}

// The menu has to say which run a file came from: three runs all ship gen_000.bin.
std::string modelLabelFor(const std::string& relative) {
    std::string path = relative;
    for (char& c : path) {
        if (c == '\\') c = '/';
    }
    const std::size_t slash = path.find_last_of('/');
    const std::string file = slash == std::string::npos ? path : path.substr(slash + 1);
    std::string parent = slash == std::string::npos ? std::string() : path.substr(0, slash);
    if (parent.size() >= 7 && parent.compare(parent.size() - 7, 7, "/models") == 0) {
        parent.resize(parent.size() - 7);
    } else if (parent == "models") {
        parent.clear();
    }
    if (parent.empty()) return file;
    const std::size_t parentSlash = parent.find_last_of('/');
    const std::string owner =
        parentSlash == std::string::npos ? parent : parent.substr(parentSlash + 1);
    return owner + "/" + file;
}

// The ruleset is a byte in the model header, so read it instead of guessing.
// Training steps from a checkpoint name: b4c32-s34760-d470705.bin -> 34760.
int modelStepsFromName(const std::string& name) {
    const std::size_t at = name.find("-s");
    if (at == std::string::npos) return 0;
    std::size_t i = at + 2;
    long long value = 0;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        value = value * 10 + (name[i] - '0');
        if (value > 1000000000LL) value = 1000000000LL;
        ++i;
    }
    return i == at + 2 ? 0 : static_cast<int>(value);
}

int readModelRuleset(const std::string& utf8Path) {
    std::FILE* file = shuyi::fopenUtf8(utf8Path, "rb");
    if (!file) return -1;
    unsigned char header[36] = {};
    const std::size_t got = std::fread(header, 1, sizeof(header), file);
    std::fclose(file);
    if (got < 12) return -1;
    // v4 keeps the v3 header layout and only grows the global vector, so the
    // rule-set byte is in the same place.
    const bool v4 = std::memcmp(header, "SHUYINN4", 8) == 0;
    const bool v3 = std::memcmp(header, "SHUYINN3", 8) == 0;
    const bool v2 = std::memcmp(header, "SHUYINN2", 8) == 0;
    if (!v3 && !v4 && !v2) return -1;
    if (v2) return 0;  // v2 has no ruleset byte; those weights are classic
    if (got < 33) return 0;
    const int ruleset = header[32];
    return ruleset < static_cast<int>(kRulesetCount) ? ruleset : -1;
}

void addModelsInDirectory(const std::string& dir, const std::string& relative,
                          std::vector<ModelEntry>& out) {
    WIN32_FIND_DATAA findData{};
    HANDLE find = FindFirstFileA((dir + "\\*.bin").c_str(), &findData);
    if (find == INVALID_HANDLE_VALUE) return;
    do {
        if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        ModelEntry entry;
        entry.path = dir + "/" + findData.cFileName;
        entry.enginePath = ansiToUtf8(entry.path);
        entry.label = widenAnsi(modelLabelFor(relative + "/" + findData.cFileName));
        entry.ruleset = readModelRuleset(entry.enginePath);
        entry.steps = modelStepsFromName(findData.cFileName);
        out.push_back(std::move(entry));
    } while (FindNextFileA(find, &findData));
    FindClose(find);
}

bool skipDirectory(const char* name) {
    return _strnicmp(name, "build", 5) == 0 || name[0] == '.' ||
           _stricmp(name, "__pycache__") == 0 || _stricmp(name, "node_modules") == 0 ||
           _stricmp(name, "archive") == 0 || _stricmp(name, "dist") == 0;
}

// Depth-limited walk: collect *.bin from every directory named "models".
void walkForModels(const std::string& dir, const std::string& relative, int depth,
                   std::vector<ModelEntry>& out) {
    const std::string base = dir.substr(dir.find_last_of("/\\") + 1);
    if (_stricmp(base.c_str(), "models") == 0) {
        addModelsInDirectory(dir, relative, out);
        return;
    }
    if (depth <= 0) return;
    WIN32_FIND_DATAA findData{};
    HANDLE find = FindFirstFileA((dir + "\\*").c_str(), &findData);
    if (find == INVALID_HANDLE_VALUE) return;
    do {
        if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (skipDirectory(findData.cFileName)) continue;
        const std::string child = relative.empty() ? std::string(findData.cFileName)
                                                   : relative + "/" + findData.cFileName;
        walkForModels(dir + "/" + findData.cFileName, child, depth - 1, out);
    } while (FindNextFileA(find, &findData));
    FindClose(find);
}

std::uint64_t modelFingerprint(const std::string& utf8Path, std::uint64_t* sizeOut) {
    // FNV-1a over the bytes: two copies of one weights file must not become two
    // menu entries that look like a choice between different networks.
    std::FILE* file = shuyi::fopenUtf8(utf8Path, "rb");
    if (!file) return 0;
    std::uint64_t hash = 1469598103934665603ULL;
    std::uint64_t total = 0;
    unsigned char chunk[65536];
    std::size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
        total += got;
        for (std::size_t i = 0; i < got; ++i) {
            hash ^= chunk[i];
            hash *= 1099511628211ULL;
        }
    }
    std::fclose(file);
    if (sizeOut) *sizeOut = total;
    return hash;
}

void scanModels() {
    std::vector<ModelEntry> found;
    for (const std::string& root : modelRoots()) walkForModels(root, std::string(), 4, found);
    std::sort(found.begin(), found.end(),
              [](const ModelEntry& a, const ModelEntry& b) { return a.label < b.label; });
    // Keep one entry per distinct file content, preferring the shallower path.
    std::vector<ModelEntry> unique;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> seen;
    const auto depthOf = [](const std::string& path) {
        return static_cast<std::size_t>(std::count(path.begin(), path.end(), '/') +
                                        std::count(path.begin(), path.end(), '\\'));
    };
    for (ModelEntry& entry : found) {
        std::uint64_t size = 0;
        const std::uint64_t print = modelFingerprint(entry.enginePath, &size);
        bool duplicate = false;
        for (std::size_t i = 0; i < seen.size(); ++i) {
            if (seen[i].first != size || seen[i].second != print) continue;
            duplicate = true;
            if (depthOf(entry.path) < depthOf(unique[i].path)) unique[i] = entry;
            break;
        }
        if (!duplicate) {
            seen.emplace_back(size, print);
            unique.push_back(std::move(entry));
        }
    }
    g_modelList = std::move(unique);
}

int modelIndexFor(const std::string& path) {
    for (std::size_t i = 0; i < g_modelList.size(); ++i) {
        if (g_modelList[i].enginePath == path || g_modelList[i].path == path) return static_cast<int>(i);
    }
    const std::string wanted = normalizedPath(path);
    for (std::size_t i = 0; i < g_modelList.size(); ++i) {
        if (normalizedPath(g_modelList[i].path) == wanted) return static_cast<int>(i);
    }
    return -1;
}

const wchar_t* rulesetLabel(Ruleset ruleset) {
    switch (ruleset) {
    case Ruleset::Blocking: return L"阻挡";
    case Ruleset::NoPass: return L"无过手";
    default: return L"现行";
    }
}

const wchar_t* modelRulesetHint(const ModelEntry& entry) {
    // Static buffer: the pointer must outlive the call that builds the label.
    static wchar_t buffer[32];
    if (entry.ruleset < 0 || entry.ruleset >= kRulesetCount) {
        MultiByteToWideChar(CP_UTF8, 0, "未知", -1, buffer, 32);
        return buffer;
    }
    return rulesetLabel(static_cast<Ruleset>(entry.ruleset));
}

// Fallback when nothing has been picked: the model with the most training steps
// trained for the rule set in force.  A model from an incompatible rule set is
// never selected silently.
std::string defaultModelPath() {
    char buffer[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableA("SHUYI_NN_MODEL", buffer, MAX_PATH);
    if (length > 0 && length < MAX_PATH) return ansiToUtf8(std::string(buffer, length));
    if (g_modelList.empty()) scanModels();
    const ModelEntry* best = nullptr;
    for (const ModelEntry& entry : g_modelList) {
        if (entry.ruleset < 0 ||
            !rulesetCompatible(static_cast<Ruleset>(entry.ruleset), g_ruleset)) {
            continue;
        }
        if (!best || entry.steps > best->steps) best = &entry;
    }
    return best ? best->enginePath : std::string();
}

void populateModelMenu() {
    if (!g_modelMenu) return;
    while (GetMenuItemCount(g_modelMenu) > 0) {
        DeleteMenu(g_modelMenu, 0, MF_BYPOSITION);
    }
    if (g_modelList.empty()) {
        AppendMenuW(g_modelMenu, MF_STRING | MF_GRAYED, 0, L"（没有找到模型文件）");
    } else {
        const int count = static_cast<int>(g_modelList.size());
        for (int i = 0; i < count && i < ID_MODEL_LIMIT - ID_MODEL_BASE; ++i) {
            const ModelEntry& entry = g_modelList[i];
            std::wstring label = entry.label;
            label += L"  [";
            label += modelRulesetHint(entry);
            label += L"]";
            UINT flags = MF_STRING;
            if (!g_nnModelPath.empty() && g_nnModelPath == entry.enginePath) flags |= MF_CHECKED;
            AppendMenuW(g_modelMenu, flags, ID_MODEL_BASE + i, label.c_str());
        }
    }
    AppendMenuW(g_modelMenu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(g_modelMenu, MF_STRING, ID_MODEL_RESCAN, L"重新扫描模型目录");
}

void syncModelChecks() {
    if (!g_modelMenu) return;
    const int index = modelIndexFor(g_nnModelPath);
    const int count = static_cast<int>(GetMenuItemCount(g_modelMenu));
    for (int position = 0; position < count; ++position) {
        const UINT id = GetMenuItemID(g_modelMenu, position);
        if (id < static_cast<UINT>(ID_MODEL_BASE) || id >= static_cast<UINT>(ID_MODEL_LIMIT)) continue;
        const bool picked =
            g_engineChoice == EngineChoice::MctsNn && static_cast<int>(id) - ID_MODEL_BASE == index;
        CheckMenuItem(g_modelMenu, id, MF_BYCOMMAND | (picked ? MF_CHECKED : MF_UNCHECKED));
    }
}

void setEngineChoice(EngineChoice choice);

void selectModel(int index) {
    if (index < 0 || index >= static_cast<int>(g_modelList.size())) return;
    const ModelEntry& entry = g_modelList[index];
    if (entry.ruleset < 0 ||
        !rulesetCompatible(static_cast<Ruleset>(entry.ruleset), g_ruleset)) {
        const std::wstring modelLabel = modelRulesetHint(entry);
        const std::wstring gameLabel = rulesetLabel(g_ruleset);
        MessageBoxW(g_hwnd,
                    (L"这个模型是用“" + modelLabel + L"规则”训练的，当前局面是“" + gameLabel +
                     L"规则”。\n\n请先在 规则 菜单切换规则（只能在新局前切换），再选模型。")
                        .c_str(),
                    L"模型与规则不匹配", MB_OK | MB_ICONINFORMATION);
        return;
    }
    g_nnModelPath = entry.enginePath;
    setEngineChoice(EngineChoice::MctsNn);
}

// ---------------------------------------------------------------------------
// Engine program setting and persistence
// ---------------------------------------------------------------------------
std::wstring settingsFilePath() {
    char overridePath[MAX_PATH * 2]{};
    const DWORD overrideLength =
        GetEnvironmentVariableA("SHUYI_UI_INI", overridePath, sizeof(overridePath));
    if (overrideLength > 0 && overrideLength < sizeof(overridePath)) {
        return widenAnsi(std::string(overridePath, overrideLength));
    }
    const std::wstring dir = engineClientDirectory();
    return dir.empty() ? std::wstring(L"shuyi-ui.ini") : dir + L"/shuyi-ui.ini";
}

// Plain `key=value` in UTF-8: the Win32 profile API reads ANSI and would mangle
// this project's folder name.
void loadSettings() {
    g_enginePath.clear();
    const std::wstring file = settingsFilePath();
    std::FILE* handle = shuyi::fopenUtf8(narrowUtf8(file), "rb");
    if (handle) {
        std::string text;
        char buffer[4096];
        for (std::size_t got = std::fread(buffer, 1, sizeof(buffer), handle); got > 0;
             got = std::fread(buffer, 1, sizeof(buffer), handle)) {
            text.append(buffer, got);
        }
        std::fclose(handle);
        std::istringstream lines(text);
        std::string line;
        while (std::getline(lines, line)) {
            if (!line.empty() && line.back() == static_cast<char>(13)) line.pop_back();
            if (line.rfind("engine=", 0) == 0) {
                g_enginePath = widenUtf8(trimAscii(line.substr(7)));
            } else if (line.rfind("analysis_visits=", 0) == 0) {
                g_analysisVisits = std::clamp(std::atoi(line.substr(16).c_str()), 0, 50000000);
            } else if (line.rfind("play_visits=", 0) == 0) {
                g_playVisits = std::clamp(std::atoi(line.substr(12).c_str()), 1, 50000000);
            } else if (line.rfind("a_value=", 0) == 0) {
                const std::string configured = trimAscii(line.substr(8));
                const char* start = configured.c_str();
                char* end = nullptr;
                const double value = std::strtod(start, &end);
                if (end != start && *end == '\0' && std::isfinite(value)) {
                    g_aValue = normalizeAValue(value);
                }
            } else if (line.rfind("ruleset=", 0) == 0) {
                Ruleset parsed = g_ruleset;
                if (parseRuleset(trimAscii(line.substr(8)), parsed)) g_ruleset = parsed;
            }
        }
    }
    // A shipped package writes `engine=shuyi_engine.exe`: a relative path is
    // resolved against this executable, not against the launch directory.
    if (!g_enginePath.empty()) {
        const bool absolute = (g_enginePath.size() > 1 && g_enginePath[1] == L':') ||
                              g_enginePath[0] == L'\\' || g_enginePath[0] == L'/';
        if (!absolute) {
            const std::wstring dir = engineClientDirectory();
            if (!dir.empty()) g_enginePath = dir + L"/" + g_enginePath;
        }
    }
    g_engineAuto = g_enginePath.empty();
}

void saveSettings() {
    const std::wstring file = settingsFilePath();
    std::FILE* handle = shuyi::fopenUtf8(narrowUtf8(file), "wb");
    if (handle) {
        std::string text = "# shuyi-ui.ini: engine=<path to any program that speaks the protocol>";
        text += static_cast<char>(10);
        text += "engine=";
        text += narrowUtf8(g_enginePath);
        text += static_cast<char>(10);
        text += "analysis_visits=" + std::to_string(g_analysisVisits);
        text += static_cast<char>(10);
        text += "play_visits=" + std::to_string(g_playVisits);
        text += static_cast<char>(10);
        text += "a_value=" + formatAValue(g_aValue);
        text += static_cast<char>(10);
        text += "ruleset=";
        text += rulesetName(g_ruleset);
        text += static_cast<char>(10);
        std::fwrite(text.data(), 1, text.size(), handle);
        std::fclose(handle);
    }
    g_engineAuto = g_enginePath.empty();
}

// The two programs are separate products, so the protocol version is the only
// compatibility promise.  A different version is refused instead of half-working.
constexpr int kUiProtocolVersion = 2;

bool checkProtocol(const EngineProbe& probe, const std::wstring& path) {
    if (!probe.answered) {
        MessageBoxW(g_hwnd, (L"这个程序没有回应 capabilities，看起来不是数弈引擎：\n" + path).c_str(),
                    L"引擎", MB_OK | MB_ICONERROR);
        return false;
    }
    if (probe.proto != std::to_string(kUiProtocolVersion)) {
        MessageBoxW(g_hwnd,
                    (L"引擎的协议版本是 " + widen(probe.proto.empty() ? "?" : probe.proto) +
                     L"，本界面需要 " + std::to_wstring(kUiProtocolVersion) +
                     L"。\n两者是独立程序，必须使用同一协议版本。")
                        .c_str(),
                    L"引擎", MB_OK | MB_ICONERROR);
        return false;
    }
    return true;
}

// The engine the user chose, else the first discovered binary that speaks the
// current protocol.  No answer means no engine: the window does not guess.
std::wstring chooseEnginePath() {
    std::wstring cpuBuild;
    for (const std::wstring& path : engineCandidates()) {
        const EngineProbe probe = probeEngine(path);
        if (!probe.answered || probe.proto != std::to_string(kUiProtocolVersion)) continue;
        if (probe.backend == "cuda") return path;
        if (cpuBuild.empty()) cpuBuild = path;
    }
    return cpuBuild;
}

bool startEngineProcess() {
    if (g_engineChoice == EngineChoice::Off) return false;
    stopEngineProcess();
    std::wstring error;
    std::wstring enginePath = g_enginePath;
    if (enginePath.empty()) enginePath = chooseEnginePath();
    if (enginePath.empty()) {
        g_engineMessage = L"没有找到可用的引擎";
        g_engineChoice = EngineChoice::Off;
        return false;
    }
    const EngineProbe probe = probeEngine(enginePath);
    if (!checkProtocol(probe, enginePath)) {
        g_engineChoice = EngineChoice::Off;
        return false;
    }
    if (!g_engineClient.start(enginePath, &error)) {
        g_engineMessage = error.empty() ? L"引擎启动失败" : error;
        g_engineChoice = EngineChoice::Off;
        return false;
    }
    g_enginePathUsed = enginePath;

    // A rejected command has to stop the start-up: otherwise the window would sit
    // on an empty analysis and the real cause would never be visible.
    std::wstring failure;
    const auto run = [&](const std::string& line) {
        if (!failure.empty()) return;
        std::string reply;
        if (!g_engineClient.command(line, &reply, 8000)) {
            failure = L"引擎没有回应 " + widen(line);
        } else if (!reply.empty() && reply[0] == '?') {
            failure = L"引擎拒绝 " + widen(line) + L"：\n" + widen(reply.substr(1));
        }
    };
    run("boardsize 9");
    run(std::string("ruleset ") + rulesetName(g_ruleset));
    run("set_a " + formatAValue(g_aValue));
    if (g_engineChoice == EngineChoice::MctsNn) {
        if (g_nnModelPath.empty()) g_nnModelPath = defaultModelPath();
        if (g_nnModelPath.empty()) {
            failure = L"没有可用模型：把 b<blocks>c<channels>-s<steps>-d<data>.bin 放进 "
                      L"models/（或引擎菜单 → MCTS + 网络 → 重新扫描）";
        } else {
            run("set_model " + g_nnModelPath);
        }
    }
    run(std::string("set_engine ") + engineChoiceName(g_engineChoice));
    run("set_backend auto");
    if (probe.visitBudget) {
        run("set_visits " + std::to_string(g_analysisVisits));
    } else {
        run("set_time 650");
    }
    if (failure.empty() && g_engineChoice == EngineChoice::MctsNn) {
        std::string capabilities;
        if (!g_engineClient.command("capabilities", &capabilities, 3000) ||
            capabilities.find("model=1") == std::string::npos) {
            failure = L"引擎没有加载到模型：\n" + widen(g_nnModelPath);
        }
    }
    if (!failure.empty()) {
        stopEngineProcess();
        g_engineChoice = EngineChoice::Off;
        g_engineMessage = failure;
        MessageBoxW(g_hwnd, failure.c_str(), L"引擎", MB_OK | MB_ICONERROR);
        return false;
    }

    g_engineReady = true;
    g_engineMessage = L"引擎已就绪";
    syncEnginePosition();
    return true;
}

// ---------------------------------------------------------------- settings ---
HWND g_settingsWnd = nullptr;
HWND g_settingsAnalysisEdit = nullptr;
HWND g_settingsPlayEdit = nullptr;
HWND g_settingsAValueEdit = nullptr;

void closeSettings(bool apply) {
    if (!g_settingsWnd) return;
    if (apply) {
        wchar_t buffer[32]{};
        const auto readInt = [&](HWND edit, int fallback, int low, int high) {
            if (!edit) return fallback;
            GetWindowTextW(edit, buffer, 32);
            wchar_t* end = nullptr;
            const long value = std::wcstol(buffer, &end, 10);
            if (end == buffer) return fallback;
            return static_cast<int>(std::clamp<long>(value, low, high));
        };
        const auto readA = [&](HWND edit, double fallback) {
            if (!edit) return fallback;
            GetWindowTextW(edit, buffer, 32);
            wchar_t* end = nullptr;
            const double value = std::wcstod(buffer, &end);
            if (end == buffer || *end != L'\0' || !std::isfinite(value)) return fallback;
            return normalizeAValue(value);
        };
        g_analysisVisits = readInt(g_settingsAnalysisEdit, g_analysisVisits, 0, 50000000);
        g_playVisits = readInt(g_settingsPlayEdit, g_playVisits, 1, 50000000);
        const double aValue = readA(g_settingsAValueEdit, g_aValue);
        if (g_engineClient.running()) sendEngine("set_visits " + std::to_string(g_analysisVisits));
        if (aValue != g_aValue) setAValue(aValue);
        else saveSettings();
    }
    EnableWindow(g_hwnd, TRUE);
    DestroyWindow(g_settingsWnd);
    g_settingsWnd = nullptr;
    g_settingsAnalysisEdit = g_settingsPlayEdit = g_settingsAValueEdit = nullptr;
    SetForegroundWindow(g_hwnd);
    refresh();
}

LRESULT CALLBACK settingsProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_ERASEBKGND: {
        RECT client{};
        GetClientRect(hwnd, &client);
        fillRectColor(reinterpret_cast<HDC>(wParam), client, kPanel);
        return 1;
    }
    case WM_COMMAND:
        if (LOWORD(wParam) == ID_SETTINGS_OK) {
            closeSettings(true);
            return 0;
        }
        if (LOWORD(wParam) == ID_SETTINGS_CANCEL || LOWORD(wParam) == IDCANCEL) {
            closeSettings(false);
            return 0;
        }
        return 0;
    case WM_CLOSE:
        closeSettings(false);
        return 0;
    default:
        break;
    }
    return DefWindowProcW(hwnd, message, wParam, lParam);
}

void openSettings() {
    if (g_settingsWnd) {
        SetForegroundWindow(g_settingsWnd);
        return;
    }
    RECT main{};
    GetWindowRect(g_hwnd, &main);
    const int width = scale(420);
    const int height = scale(210);
    const int x = main.left + ((main.right - main.left) - width) / 2;
    const int y = main.top + ((main.bottom - main.top) - height) / 2;
    g_settingsWnd = CreateWindowExW(WS_EX_DLGMODALFRAME, kSettingsClass, L"设置",
                                    WS_POPUP | WS_CAPTION | WS_SYSMENU, x, y, width, height, g_hwnd,
                                    nullptr, g_instance, nullptr);
    if (!g_settingsWnd) return;

    const auto label = [&](const wchar_t* text, int top) {
        CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE | SS_LEFT, scale(16), top,
                        scale(170), scale(20), g_settingsWnd, nullptr, g_instance, nullptr);
    };
    const auto editText = [&](int top, const std::wstring& text) {
        HWND box = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", text.c_str(),
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL, scale(196),
                                   top - scale(2), scale(150), scale(24), g_settingsWnd, nullptr,
                                   g_instance, nullptr);
        SendMessageW(box, WM_SETFONT, reinterpret_cast<WPARAM>(g_fontUi), TRUE);
        return box;
    };
    const auto edit = [&](int top, int value) {
        wchar_t text[24]{};
        std::swprintf(text, 24, L"%d", value);
        return editText(top, text);
    };

    const int rowHeight = scale(34);
    int top = scale(20);
    label(L"分析访问上限（0 = 不限）", top);
    g_settingsAnalysisEdit = edit(top, g_analysisVisits);
    top += rowHeight;
    label(L"落子访问上限", top);
    g_settingsPlayEdit = edit(top, g_playVisits);
    top += rowHeight;
    label(L"A值（步长 0.5）", top);
    g_settingsAValueEdit = editText(top, formatAValueW(g_aValue));

    const int buttonTop = height - scale(52);
    HWND ok = CreateWindowExW(0, L"BUTTON", L"确定",
                              WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON, scale(196),
                              buttonTop, scale(72), scale(28), g_settingsWnd,
                              reinterpret_cast<HMENU>(static_cast<INT_PTR>(ID_SETTINGS_OK)),
                              g_instance, nullptr);
    HWND cancel = CreateWindowExW(0, L"BUTTON", L"取消", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                                  scale(274), buttonTop, scale(72), scale(28), g_settingsWnd,
                                  reinterpret_cast<HMENU>(static_cast<INT_PTR>(ID_SETTINGS_CANCEL)),
                                  g_instance, nullptr);
    for (HWND child : {ok, cancel}) {
        SendMessageW(child, WM_SETFONT, reinterpret_cast<WPARAM>(g_fontUi), TRUE);
    }
    for (HWND child : {g_settingsAnalysisEdit, g_settingsPlayEdit, g_settingsAValueEdit}) {
        SendMessageW(child, EM_SETSEL, 0, -1);
    }

    EnableWindow(g_hwnd, FALSE);
    ShowWindow(g_settingsWnd, SW_SHOW);
    SetForegroundWindow(g_settingsWnd);
    SetFocus(g_settingsAnalysisEdit);
}

// 选择引擎程序…: a plain file dialog.  Any program that speaks the protocol is
// acceptable -- it does not have to be built from this repository.
void pickEngineProgram() {
    wchar_t file[MAX_PATH * 2]{};
    if (!g_enginePathUsed.empty()) wcsncpy(file, g_enginePathUsed.c_str(), MAX_PATH * 2 - 1);
    std::wstring filter;
    filter += L"引擎程序 (*.exe)";
    filter.push_back(0);
    filter += L"*.exe";
    filter.push_back(0);
    filter += L"所有程序 (*.*)";
    filter.push_back(0);
    filter += L"*.*";
    filter.push_back(0);
    filter.push_back(0);

    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = g_hwnd;
    dialog.lpstrFilter = filter.c_str();
    dialog.lpstrFile = file;
    dialog.nMaxFile = MAX_PATH * 2;
    dialog.lpstrTitle = L"选择引擎程序";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetOpenFileNameW(&dialog)) return;

    const std::wstring chosen(file);
    const EngineProbe probe = probeEngine(chosen);
    if (!checkProtocol(probe, chosen)) return;
    g_enginePath = chosen;
    saveSettings();
    if (g_engineChoice != EngineChoice::Off) {
        const EngineChoice keep = g_engineChoice;
        g_engineChoice = EngineChoice::Off;
        setEngineChoice(keep);
    } else {
        refresh(true);
    }
}

void useAutoEngine() {
    g_enginePath.clear();
    saveSettings();
    if (g_engineChoice != EngineChoice::Off) {
        const EngineChoice keep = g_engineChoice;
        g_engineChoice = EngineChoice::Off;
        setEngineChoice(keep);
    } else {
        refresh(true);
    }
}

void showEngineInfo() {
    std::wstring text = L"引擎程序：" + (g_enginePathUsed.empty() ? std::wstring(L"（未启动）")
                                                                 : g_enginePathUsed);
    text += L"\n\n设置：";
    text += g_enginePath.empty() ? L"自动探测" : (L"手动指定（" + settingsFilePath() + L"）");
    if (!g_enginePathUsed.empty()) {
        const EngineProbe probe = probeEngine(g_enginePathUsed);
        text += L"\n\n协议版本：" + (probe.proto.empty() ? std::wstring(L"未知（旧引擎）")
                                                        : widen(probe.proto));
        text += L"\n推理后端：" + (probe.backend == "cuda" ? std::wstring(L"CUDA")
                                                          : std::wstring(L"CPU"));
    }
    text += L"\n\n模型：" + (g_nnModelPath.empty() ? std::wstring(L"（未设置）")
                                                    : fileNameOf(g_nnModelPath));
    MessageBoxW(g_hwnd, text.c_str(), L"引擎信息", MB_OK | MB_ICONINFORMATION);
}

void setEngineChoice(EngineChoice choice) {
    g_analysisOn = false;
    stopEngineProcess();
    g_engineChoice = choice;
    if (g_engineChoice != EngineChoice::Off) {
        if (!startEngineProcess()) {
            MessageBoxW(g_hwnd, g_engineMessage.c_str(), L"引擎", MB_OK | MB_ICONWARNING);
        }
    }
    syncEngineChecks();
    noteComputerToMove();
    refresh();
}

void handleEngineLine(const std::string& raw) {
    std::string line = raw;
    std::string commandId;
    const std::size_t space = line.find(' ');
    if (space != std::string::npos && space <= 9) {
        bool numeric = true;
        for (std::size_t i = 0; i < space; ++i) {
            if (line[i] < '0' || line[i] > '9') {
                numeric = false;
                break;
            }
        }
        if (numeric) {
            commandId = line.substr(0, space);
            line = line.substr(space + 1);
        }
    }
    if (line.rfind("info ", 0) == 0) {
        if ((g_analysisOn || g_enginePending) && g_analysisNode >= 0) {
            applyEngineAnalysis(g_analysisNode, line.substr(5));
        }
        return;
    }
    if (line.rfind("= done", 0) == 0) {
        // The analysis ran out of budget on its own.  The numbers stay on screen
        // and the tree stays in the engine, so Space can extend it.
        g_analysisOn = false;
        g_analysisFinished = true;
        g_engineMessage = L"已达访问上限";
        refresh();
        return;
    }
    if (line.rfind("= stopped", 0) == 0) return;
    if (!line.empty() && line[0] == '?') {
        const bool wasPending = g_enginePending;
        g_engineMessage = L"引擎错误：" + widen(line.substr(1));
        g_enginePending = false;
        g_pendingReplyId.clear();
        g_thinking = false;
        if (wasPending) refresh();
        return;
    }
    if (line.rfind("= ", 0) != 0) return;
    const std::string payload = line.substr(2);
    if (!g_enginePending) return;
    // An empty payload is an acknowledgement of some earlier command, and a
    // different id belongs to another command: neither is the move we await.
    if (payload.empty()) return;
    if (!g_pendingReplyId.empty() && commandId != g_pendingReplyId) return;
    if (payload == "pass") {
        g_enginePending = false;
        g_pendingReplyId.clear();
        g_engineMessage = L"引擎过手";
        g_thinking = false;
        recordMove(Move(kPassCell, kPassNumber), g_position.sideToMove());
        refresh();
        return;
    }
    std::istringstream input(payload);
    std::string vertex;
    int number = 0;
    input >> vertex >> number;
    int cell = 0;
    if (!parseSquare(vertex, cell) || number < 1 || number > 9) {
        g_enginePending = false;
        g_pendingReplyId.clear();
        g_engineMessage = L"引擎返回了无法识别的着法：" + widen(payload);
        refresh();
        return;
    }
    const Move move(cell, number);
    if (!g_position.isLegal(move)) {
        g_enginePending = false;
        g_pendingReplyId.clear();
        g_engineMessage = L"引擎返回非法着法：" + widen(payload);
        refresh();
        return;
    }
    g_enginePending = false;
    g_pendingReplyId.clear();
    g_thinking = false;
    recordMove(move, g_position.sideToMove());
    g_engineMessage = L"引擎已落子";
    refresh();
    if (aiToMove()) maybeAiMove();
}

void drainEngineLines() {
    for (const std::string& line : g_engineClient.drain()) handleEngineLine(line);
}

void syncRulesetChecks() {
    if (!g_rulesetMenu) return;
    int id = ID_RULESET_CLASSIC;
    if (g_ruleset == Ruleset::NoPass) id = ID_RULESET_NOPASS;
    else if (g_ruleset == Ruleset::Blocking) id = ID_RULESET_BLOCKING;
    CheckMenuRadioItem(g_rulesetMenu, ID_RULESET_CLASSIC, ID_RULESET_NOPASS, id, MF_BYCOMMAND);
}

// Switching the rule set is only legal before the first stone: the two games have
// incompatible move generators and incompatible value heads.
bool setRuleset(Ruleset ruleset) {
    if (ruleset == g_ruleset) return true;
    if (g_position.stones() > 0) {
        MessageBoxW(g_hwnd, L"换规则需要先开新局（规则不能在一盘棋中途切换）。",
                    L"切换规则", MB_OK | MB_ICONINFORMATION);
        syncRulesetChecks();
        return false;
    }
    g_ruleset = ruleset;
    rebuildPosition();
    if (g_engineChoice != EngineChoice::Off) {
        g_analysisOn = false;
        stopEngineProcess();
        if (!startEngineProcess()) {
            MessageBoxW(g_hwnd, g_engineMessage.c_str(), L"引擎", MB_OK | MB_ICONWARNING);
        }
    }
    syncRulesetChecks();
    refresh(true);
    return true;
}

void syncEngineChecks() {
    if (!g_engineMenu) return;
    int id = ID_ENGINE_OFF;
    if (g_engineChoice == EngineChoice::Mcts) id = ID_ENGINE_MCTS;
    CheckMenuRadioItem(g_engineMenu, ID_ENGINE_OFF, ID_ENGINE_MCTS, id, MF_BYCOMMAND);
    if (g_modelMenuPosition >= 0) {
        CheckMenuItem(g_engineMenu, static_cast<UINT>(g_modelMenuPosition),
                      MF_BYPOSITION |
                          (g_engineChoice == EngineChoice::MctsNn ? MF_CHECKED : MF_UNCHECKED));
    }
    syncModelChecks();
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
struct Layout {
    RECT client{};
    RECT left{};     // the analysis panel
    RECT board{};
    RECT right{};  // the record tree
    RECT bottom{}; // A value and the pieces in hand
    int cell = 60;
};

Layout computeLayout(HWND hwnd) {
    Layout result;
    GetClientRect(hwnd, &result.client);
    const int margin = scale(16);
    const int gap = scale(14);
    const int leftWidth = scale(292);
    const int rightWidth = scale(246);
    const int barHeight = scale(64);
    const int coord = scale(24);

    result.left = {margin, margin, margin + leftWidth,
                   result.client.bottom - barHeight - scale(8)};
    result.bottom = {0, result.client.bottom - barHeight, result.client.right, result.client.bottom};
    result.right = {result.client.right - margin - rightWidth, result.left.top,
                    result.client.right - margin, result.left.bottom};

    const int slotLeft = result.left.right + gap;
    const int slotWidth = result.right.left - gap - slotLeft;
    const int slotHeight = result.left.bottom - result.left.top;
    int cell = std::min((slotWidth - 2 * coord) / 9, (slotHeight - 2 * coord) / 9);
    cell = std::max(scale(26), std::min(cell, scale(72)));
    const int size = cell * 9;
    const int boardLeft = slotLeft + (slotWidth - size) / 2;
    const int boardTop = result.left.top + (slotHeight - size) / 2;
    result.board = {boardLeft, boardTop, boardLeft + size, boardTop + size};
    result.cell = cell;
    return result;
}

struct Chrome {
    RECT aMinus{};
    RECT aPlus{};
    RECT chips[2][9];  // [red, black][1..9]
};

Chrome computeChrome(const Layout& layout) {
    Chrome chrome;
    // Bottom bar: A value on the left, then the two rows of pieces.
    const int barTop = layout.bottom.top;
    const int aRow = barTop + scale(21);
    chrome.aMinus = {scale(96), aRow, scale(118), aRow + scale(21)};
    chrome.aPlus = {scale(122), aRow, scale(144), aRow + scale(21)};
    int cursor = scale(168);
    const int labelWidth = scale(20);
    const int chipWidth = scale(30);
    const int chipGap = scale(2);
    const int rowTop = barTop + scale(11);
    for (int side = 0; side < 2; ++side) {
        const int row = rowTop + side * scale(24);
        const int first = cursor + labelWidth;
        for (int n = 0; n < 9; ++n) {
            chrome.chips[side][n] = {first + n * (chipWidth + chipGap), row,
                                     first + n * (chipWidth + chipGap) + chipWidth,
                                     row + scale(21)};
        }
    }
    return chrome;
}

bool contains(const RECT& rect, POINT point) {
    return point.x >= rect.left && point.x < rect.right && point.y >= rect.top &&
           point.y < rect.bottom;
}

// The record tree's geometry, shared by the drawing and the hit test.
//
// It is laid out like a score sheet, not like a wide diagram: one row per move,
// the trunk continuing straight down and an alternative starting an indented
// block right after the move it replaces.  A 54-move game has 54 columns if you
// put the plies on the x axis, and this panel is 246 px wide -- a column tree
// showed five moves and hid the rest.  Vertically there is room to scroll, and
// the row order (move, its alternatives, then the next move of the trunk) is the
// order a player reads a record in.
struct TreeCell {
    int node = 0;
    RECT box{};
    int level = 0;       // how deep the variation nesting is, for the indent
    bool variation = false;
};
struct TreeLayout {
    std::vector<TreeCell> cells;
    int rowHeight = 19;
    int contentHeight = 0;
};

TreeLayout computeTreeLayout(const Layout& layout, int scroll) {
    TreeLayout tree;
    tree.rowHeight = scale(20);
    const int left = layout.right.left + scale(10);
    const int width = (layout.right.right - layout.right.left) - scale(20);
    const int top = layout.right.top + scale(34);
    const int indent = scale(12);
    int row = 0;
    const auto emit = [&](int id, int level, bool variation) {
        TreeCell cell;
        cell.node = id;
        cell.level = level;
        cell.variation = variation;
        const int x = left + level * indent;
        const int y = top + row * tree.rowHeight - scroll;
        cell.box = {x, y, x + width - level * indent, y + tree.rowHeight - scale(2)};
        tree.cells.push_back(cell);
        ++row;
    };
    // PGN order: the move, then its alternatives indented right after it, then
    // the move's own continuation.  That is exactly where a reader looks for
    // "what else could he have played here", and the trunk stays contiguous.
    std::function<void(int, int)> walkRest = [&](int id, int level) {
        const auto& children = g_record.childrenOf(id);
        if (children.empty()) return;
        emit(children.front(), level, false);
        for (std::size_t i = 1; i < children.size(); ++i) {
            emit(children[i], level + 1, true);
            walkRest(children[i], level + 1);
        }
        walkRest(children.front(), level);
    };
    emit(g_record.root(), 0, false);
    walkRest(g_record.root(), 0);
    tree.contentHeight = row * tree.rowHeight;
    return tree;
}

// Where the tree has to be scrolled so `id` is visible.
int treeScrollFor(int id) {
    Layout layout = computeLayout(g_hwnd);
    const TreeLayout tree = computeTreeLayout(layout, 0);
    const int visibleTop = layout.right.top + scale(34);
    const int visibleHeight = layout.right.bottom - visibleTop;
    for (const TreeCell& cell : tree.cells) {
        if (cell.node != id) continue;
        const int nodeTop = cell.box.top - layout.right.top - scale(34);
        if (nodeTop < g_treeScroll) return nodeTop;
        if (nodeTop + tree.rowHeight > g_treeScroll + visibleHeight - tree.rowHeight) {
            return std::max(0, nodeTop - visibleHeight + 3 * tree.rowHeight);
        }
        return g_treeScroll;
    }
    return g_treeScroll;
}

// ---------------------------------------------------------------------------
// Board drawing
// ---------------------------------------------------------------------------
void drawStone(HDC dc, int cx, int cy, int radius, bool red, int number, bool ghost = false) {
    COLORREF fill = red ? kRed : kBlack;
    COLORREF border = red ? RGB(122, 24, 26) : RGB(8, 9, 12);
    COLORREF text = RGB(255, 245, 232);
    if (ghost) {
        fill = red ? RGB(219, 156, 139) : RGB(124, 127, 136);
        border = kBoardLine;
        text = RGB(248, 242, 228);
    }
    const RECT body{cx - radius, cy - radius, cx + radius, cy + radius};
    const int corner = std::max(2, radius / 4);
    roundRect(dc, body, corner, fill, border, std::max(1, scale(2)));
    drawTextAt(dc, std::to_wstring(number), cx - radius, cy - radius, radius * 2, radius * 2,
               g_fontPiece, text);
}

// One mark per analysed move, drawn with the same square footprint as a piece.
//
//   row 1  win rate
//   row 2  visits, abbreviated to at most four characters
//   row 3  expected terminal score
//
// Colour is relative to the best move, not absolute.  Marks more than 20 points
// behind, or with under 5% of the best move's visits, are not drawn at all --
// except a solved one, which is a fact and is always drawn.
// On a converged opening search every one of the ~40 legal placements comes back
// at ~40%, and drawing all of them turns the board into a wall of numbers.  Six
// marks always, then only moves that are close to the best and were really
// searched: the board should answer "what are the candidates", not "what did the
// engine touch".
constexpr double kMarkWinRateWindow = 0.10;
constexpr double kMarkVisitFloor = 0.20;
constexpr int kMarkAlwaysShow = 6;

// A move that beat the number the previous position gave it by this much is
// marked as a find (blue) rather than as a good move (green): half the width of
// the board's red-to-green ramp, which is where a rise stops being noise.
constexpr double kRowGainWindow = 0.05;

COLORREF markColor(double relative, bool best) {
    if (best) return RGB(126, 196, 255);  // light blue
    const double t = std::clamp(1.0 + relative / kMarkWinRateWindow, 0.0, 1.0);
    return mixColor(RGB(214, 92, 82), RGB(96, 186, 118), t);
}

void drawAnalysisMarks(HDC dc, const Layout& layout, const std::vector<int>& rankedCells) {
    const auto& analysis = currentAnalysis();
    if (rankedCells.empty()) return;
    const double bestRate = displayWinRate(analysis.cells[rankedCells.front()]);
    const int cell = layout.cell;
    const RECT& board = layout.board;

    int mostVisits = 0;
    for (int index : rankedCells) mostVisits = std::max(mostVisits, analysis.cells[index].visits);

    const int radius = static_cast<int>(cell * 0.40);
    const int corner = std::max(2, radius / 4);
    const int rowHeight = std::max(8, (radius * 2) / 3);
    const int textTop = -radius;
    for (std::size_t rank = 0; rank < rankedCells.size(); ++rank) {
        const int cellIndex = rankedCells[rank];
        const auto& info = analysis.cells[cellIndex];
        const bool topMark = rank < kMarkAlwaysShow;
        const double rate = displayWinRate(info);
        const double relative = rate - bestRate;
        if (!topMark && !info.proven && relative < -kMarkWinRateWindow) continue;
        if (!topMark && !info.proven && info.visits < static_cast<int>(mostVisits * kMarkVisitFloor)) {
            continue;
        }
        const COLORREF colour = markColor(relative, rank == 0);
        const int cx = board.left + xOf(cellIndex) * cell + cell / 2;
        const int cy = board.top + (8 - yOf(cellIndex)) * cell + cell / 2;
        const RECT mark{cx - radius, cy - radius, cx + radius, cy + radius};
        roundRect(dc, mark, corner, mixColor(kBoard, colour, 0.62), colour, std::max(1, scale(1)));
        // Which piece to play is half of the move: the same cell takes nine
        // numbers, and a mark that says "42%" says nothing about which of them.
        // The number sits on the top-right corner, raised, the way the count sits
        // on the chips: a superscript, not a fourth line.
        const int dot = std::max(scale(6), radius / 3);
        const int dotX = cx + radius - dot / 2;
        const int dotY = cy - radius + dot / 2;
        filledCircle(dc, dotX, dotY, dot, RGB(28, 32, 38));
        drawTextAt(dc, std::to_wstring(info.move.number), dotX - dot, dotY - dot, dot * 2, dot * 2,
                   g_fontTiny, RGB(255, 245, 232));
        int y = cy + textTop;
        drawTextAt(dc, formatWinRate(rate), cx - radius, y, radius * 2, rowHeight, g_fontSmall,
                   RGB(28, 32, 38));
        y += rowHeight;
        if (info.visits > 0) {
            drawTextAt(dc, formatVisits(info.visits), cx - radius, y, radius * 2, rowHeight,
                       g_fontSmall, RGB(28, 32, 38));
        }
        y += rowHeight;
        std::wstring scoreText = L"--";
        if (info.hasScore) {
            scoreText = info.proven ? formatPoints(info.averageScore)
                                    : formatAverageScore(info.averageScore);
        }
        drawTextAt(dc, scoreText, cx - radius, y, radius * 2, rowHeight, g_fontSmall,
                   RGB(28, 32, 38));
    }
}

void drawBoard(HDC dc, const Layout& layout, const std::vector<int>& rankedCells) {
    const int cell = layout.cell;
    const RECT& board = layout.board;

    fillRectColor(dc, board, kBoard);
    for (int i = 0; i < scale(5); ++i) {
        RECT edge{board.left + i, board.top + i, board.right - i, board.bottom - i};
        HBRUSH brush = CreateSolidBrush(i == 0 ? kBoardLine : kBoard);
        FrameRect(dc, &edge, brush);
        DeleteObject(brush);
    }
    for (int i = 0; i <= 9; ++i) {
        const int x = board.left + i * cell;
        const int y = board.top + i * cell;
        line(dc, x, board.top, x, board.bottom, kBoardLine, i == 0 || i == 9 ? 2 : 1);
        line(dc, board.left, y, board.right, y, kBoardLine, i == 0 || i == 9 ? 2 : 1);
    }

    drawAnalysisMarks(dc, layout, rankedCells);

    const int stars[5][2] = {{2, 2}, {6, 2}, {4, 4}, {2, 6}, {6, 6}};
    for (const auto& star : stars) {
        const int cx = board.left + star[0] * cell + cell / 2;
        const int cy = board.top + (8 - star[1]) * cell + cell / 2;
        filledCircle(dc, cx, cy, std::max(2, scale(3)), kBoardLine);
    }

    for (int i = 0; i < 9; ++i) {
        const std::wstring letter(1, static_cast<wchar_t>(L'a' + i));
        const std::wstring number = std::to_wstring(9 - i);
        const int cx = board.left + i * cell + cell / 2;
        drawTextAt(dc, letter, cx - cell / 2, board.top - scale(24), cell, scale(20), g_fontCoord,
                   kMuted);
        drawTextAt(dc, letter, cx - cell / 2, board.bottom + scale(4), cell, scale(20), g_fontCoord,
                   kMuted);
        const int cy = board.top + i * cell + cell / 2;
        drawTextAt(dc, number, board.left - scale(24), cy - cell / 2, scale(20), cell, g_fontCoord,
                   kMuted);
        drawTextAt(dc, number, board.right + scale(4), cy - cell / 2, scale(20), cell, g_fontCoord,
                   kMuted);
    }

    if (g_hoverCell >= 0 && humanToMove() && !g_thinking && g_position.stoneAt(g_hoverCell) == 0 &&
        g_position.remaining(g_position.sideToMove(), g_selected) > 0) {
        const int x = xOf(g_hoverCell);
        const int y = yOf(g_hoverCell);
        const int cx = board.left + x * cell + cell / 2;
        const int cy = board.top + (8 - y) * cell + cell / 2;
        drawStone(dc, cx, cy, static_cast<int>(cell * 0.36), g_position.sideToMove() == Player::Red,
                  g_selected, true);
    }

    // The move that reached this node: a wide, heavy ring, because the thin one
    // read as part of the piece.
    const Move last = g_record.current() > 0 ? g_position.lastMove() : Move{};
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 9; ++x) {
            const int cellIndex = index(x, y);
            const int stone = g_position.stoneAt(cellIndex);
            if (stone == 0) continue;
            const int cx = board.left + x * cell + cell / 2;
            const int cy = board.top + (8 - y) * cell + cell / 2;
            const int radius = static_cast<int>(cell * 0.37);
            drawStone(dc, cx, cy, radius, stone > 0, stone > 0 ? stone : -stone);
            if (!isPass(last) && last.valid() && last.cell == cellIndex) {
                const int grow = scale(6);
                const RECT ring{cx - radius - grow, cy - radius - grow, cx + radius + grow,
                                cy + radius + grow};
                HPEN pen = CreatePen(PS_SOLID, std::max(scale(3), scale(4)), kGold);
                HGDIOBJ old = SelectObject(dc, pen);
                HGDIOBJ oldBrush = SelectObject(dc, GetStockObject(NULL_BRUSH));
                RoundRect(dc, ring.left, ring.top, ring.right, ring.bottom, std::max(4, radius / 2),
                          std::max(4, radius / 2));
                SelectObject(dc, oldBrush);
                SelectObject(dc, old);
                DeleteObject(pen);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Left panel: everything the analysis produced
// ---------------------------------------------------------------------------
// The chart is always Red's view: the engine reports from the side to move, and a
// curve that flips perspective every ply is unreadable.
double redWinRate(const EngineAnalysisView& view, int depth) {
    return (depth % 2 == 0) ? view.winrate : 1.0 - view.winrate;
}

double redScore(const EngineAnalysisView& view, int depth) {
    return (depth % 2 == 0) ? view.averageScore : -view.averageScore;
}

void drawChart(HDC dc, const RECT& box) {
    if (box.bottom - box.top < scale(60)) return;
    const int node = g_record.current();
    const std::vector<int> path = g_record.line(node);
    const int width = box.right - box.left;
    roundRect(dc, box, scale(8), kPanelDeep);

    int analysed = 0;
    for (int id : path) {
        if (g_record.node(id).analysis.valid) ++analysed;
    }
    if (analysed == 0) {
        const wchar_t* message = path.empty() ? L"走势：还没有着法" : L"走势：这条线还没有分析";
        drawTextAt(dc, message, box.left + scale(8), box.top, width - scale(16),
                   box.bottom - box.top, g_fontSmall, kMuted);
        return;
    }

    const int padLeft = scale(8);
    const int padRight = scale(8);
    const int plotTop = box.top + scale(14);
    const int plotBottom = box.bottom - scale(16);
    const int plotLeft = box.left + padLeft;
    const int plotRight = box.right - padRight;
    if (plotBottom - plotTop < scale(20)) return;
    const int columns = static_cast<int>(path.size());
    const double stepX = columns > 1 ? static_cast<double>(plotRight - plotLeft) / (columns - 1) : 0.0;
    const auto xAt = [&](int index) {
        return plotLeft + static_cast<int>(std::lround(index * stepX));
    };
    const auto yWin = [&](double rate) {
        return plotBottom - static_cast<int>(std::lround((plotBottom - plotTop) * std::clamp(rate, 0.0, 1.0)));
    };

    // Score curve: its own scale, symmetric around 0.
    double maxScore = 2.0;
    for (int id : path) {
        const auto& view = g_record.node(id).analysis;
        if (!view.valid || !view.hasScore) continue;
        maxScore = std::max(maxScore, std::fabs(redScore(view, g_record.depth(id))));
    }
    const auto yScore = [&](double score) {
        const double t = std::clamp(score / maxScore, -1.0, 1.0);
        return plotBottom - static_cast<int>(std::lround((plotBottom - plotTop) * (t + 1.0) / 2.0));
    };

    line(dc, plotLeft, yWin(0.5), plotRight, yWin(0.5), RGB(58, 66, 80));
    line(dc, plotLeft, yScore(0.0), plotRight, yScore(0.0), RGB(48, 54, 66));

    // A segment is drawn only between two *analysed* points: a gap is the honest
    // rendering of "the engine never looked at this position".
    int previous = -1;
    for (int i = 0; i < columns; ++i) {
        const auto& view = g_record.node(path[i]).analysis;
        if (!view.valid) {
            previous = -1;
            continue;
        }
        if (previous >= 0) {
            const auto& before = g_record.node(path[previous]).analysis;
            line(dc, xAt(previous), yWin(redWinRate(before, g_record.depth(path[previous]))),
                 xAt(i), yWin(redWinRate(view, g_record.depth(path[i]))), kCyan, 2);
            if (before.hasScore && view.hasScore) {
                line(dc, xAt(previous), yScore(redScore(before, g_record.depth(path[previous]))),
                     xAt(i), yScore(redScore(view, g_record.depth(path[i]))), kGold, 2);
            }
        }
        previous = i;
    }
    for (int i = 0; i < columns; ++i) {
        const auto& view = g_record.node(path[i]).analysis;
        if (!view.valid) continue;
        const bool current = path[i] == node;
        const int cx = xAt(i);
        filledCircle(dc, cx, yWin(redWinRate(view, g_record.depth(path[i]))),
                     current ? scale(4) : scale(3), kCyan);
        if (view.hasScore) {
            filledCircle(dc, cx, yScore(redScore(view, g_record.depth(path[i]))),
                         current ? scale(4) : scale(3), kGold);
        }
    }

    // Axis labels, minimal: 100% / 0% for the win rate, the score range on the
    // right.  Anything more is noise in a 200 px box.
    drawTextAt(dc, L"100%", box.left + scale(4), plotTop - scale(12), scale(34), scale(14),
               g_fontSmall, kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, L"0%", box.left + scale(4), plotBottom, scale(30), scale(14), g_fontSmall,
               kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    if (maxScore > 0.0) {
        drawTextAt(dc, signedDecimal(maxScore), box.right - scale(44), plotTop - scale(12),
                   scale(40), scale(14), g_fontSmall, RGB(150, 126, 70),
                   DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    }
}

void drawLeftPanel(HDC dc, const Layout& layout, const std::vector<int>& rankedCells) {
    const RECT& box = layout.left;
    roundRect(dc, box, scale(12), kPanel);
    const int x = box.left + scale(14);
    const int width = (box.right - box.left) - scale(28);
    int y = box.top + scale(10);
    const auto& analysis = currentAnalysis();

    // 行棋方
    const Player side = g_position.sideToMove();
    const COLORREF sideColour = side == Player::Red ? kRed : RGB(15, 17, 22);
    filledCircle(dc, x + scale(8), y + scale(11), scale(6), sideColour,
                 side == Player::Red ? RGB(244, 164, 150) : kBlackSoft, 1);
    std::wstring turn = side == Player::Red ? L"红方行棋" : L"黑方行棋";
    if (g_position.gameOver()) turn = std::wstring(L"终局 · ") + outcomeText(g_position.outcome(g_aValue));
    else if (g_thinking) turn = L"电脑思考中…";
    drawTextAt(dc, turn, x + scale(22), y, width - scale(22), scale(24), g_fontTurn, kText,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    y += scale(26);

    // The headline of the best line, then the bar that splits the board between
    // the two sides.
    double rate = 0.5;
    bool hasRate = false;
    std::wstring scoreText = L"--";
    if (analysis.valid) {
        // The engine's own choice, in its own terms: a solved move carries its
        // exact rate, an estimated one the confidence bound it is ranked by.
        const int bestCell = analysis.best.cell;
        if (analysis.hasCell[bestCell]) {
            const auto& info = analysis.cells[bestCell];
            rate = displayWinRate(info);
        } else {
            rate = analysis.proven ? analysis.winrate
                                   : winRateLowerBound(analysis.winrate, analysis.visits);
        }
        hasRate = true;
        if (analysis.hasScore) scoreText = signedDecimal(analysis.averageScore);
    }
    const double redRate = (side == Player::Red || g_position.gameOver()) ? rate : 1.0 - rate;
    drawTextAt(dc, hasRate ? L"胜率 " + formatWinRate(redRate) : L"胜率 --", x, y,
               width / 2, scale(20), g_fontUi, hasRate ? kText : kMuted,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, L"目差 " + scoreText, x + width / 2, y, width / 2, scale(20), g_fontUi,
               analysis.hasScore ? kText : kMuted, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    y += scale(22);

    RECT bar{x, y, x + width, y + scale(14)};
    roundRect(dc, bar, scale(7), RGB(20, 22, 27));  // black side
    RECT redPart = bar;
    redPart.right = bar.left + static_cast<int>(std::lround((bar.right - bar.left) * std::clamp(redRate, 0.0, 1.0)));
    if (redPart.right > redPart.left) roundRect(dc, redPart, scale(7), kRed);
    y += scale(14) + scale(10);

    // AI 评估: how much the search has spent, and how fast.
    drawTextAt(dc, L"AI评估", x, y, scale(56), scale(18), g_fontSmall, kMuted,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    std::wstring engineLine = L"--";
    if (analysis.valid) {
        engineLine = analysisEffortText(analysis);
    } else if (g_engineChoice != EngineChoice::Off) {
        engineLine = g_analysisOn ? L"搜索中…" : L"空格开始分析";
    }
    drawTextAt(dc, engineLine, x + scale(56), y, width - scale(56), scale(18), g_fontSmall,
               g_engineChoice == EngineChoice::Off ? kMuted : kCyan,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    y += scale(20);
    if (g_engineChoice != EngineChoice::Off) {
        drawTextAt(dc, g_engineMessage, x + scale(56), y, width - scale(56), scale(16), g_fontSmall,
                   kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        y += scale(18);
    }
    y += scale(4);
    line(dc, x, y, x + width, y, RGB(58, 66, 80));
    y += scale(8);

    // 选点: one row per distinct move.  A solved move is a fact, so it is marked
    // and never hidden by the visit floor.
    drawTextAt(dc, L"选点", x, y, scale(56), scale(18), g_fontSmall, kMuted,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    y += scale(20);
    if (analysis.valid && !rankedCells.empty()) {
        const std::vector<int> rows = distinctSymmetricCells(rankedCells, analysis, g_position);
        const int count = std::min<int>(5, static_cast<int>(rows.size()));
        const double topRate = displayWinRate(analysis.cells[rows.front()]);
        for (int i = 0; i < count; ++i) {
            const auto& info = analysis.cells[rows[i]];
            const double value = displayWinRate(info);
            const int rowY = y + i * scale(23);
            const std::wstring label = moveLabel(info.move);
            drawTextAt(dc, label, x, rowY, scale(46), scale(20), g_fontUi,
                       i == 0 ? kText : kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
            if (info.proven) {
                drawTextAt(dc, L"解", x + scale(42), rowY, scale(16), scale(20), g_fontSmall, kGold,
                           DT_LEFT | DT_VCENTER | DT_SINGLELINE);
            }
            RECT track{x + scale(60), rowY + scale(7), x + width - scale(88), rowY + scale(15)};
            roundRect(dc, track, scale(3), RGB(55, 62, 75));
            RECT fill = track;
            fill.right = fill.left + static_cast<int>(std::lround((track.right - track.left) *
                                                                std::clamp(value, 0.0, 1.0)));
            if (fill.right > fill.left) {
                roundRect(dc, fill, scale(3), i == 0 ? kCyan : RGB(76, 140, 176));
            }
            drawTextAt(dc, formatWinRate(value), x + width - scale(86), rowY, scale(40), scale(20),
                       g_fontSmall, i == 0 ? kText : kMuted,
                       DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
            std::wstring score = L"--";
            if (info.hasScore) score = signedDecimal(info.averageScore);
            if (info.proven && info.hasScore) score = formatPoints(info.averageScore);
            drawTextAt(dc, score, x + width - scale(44), rowY, scale(44), scale(20), g_fontSmall,
                       info.proven ? kGold : kMuted, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
        }
        (void)topRate;
        y += count * scale(23);
    } else {
        drawTextAt(dc, L"还没有分析", x + scale(56), y, width - scale(56), scale(20), g_fontSmall,
                   kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        y += scale(23);
    }
    y += scale(4);
    line(dc, x, y, x + width, y, RGB(58, 66, 80));
    y += scale(6);

    // 当前分值: what the board is worth if the game ended now.
    drawTextAt(dc, L"当前分值", x, y, scale(76), scale(22), g_fontSmall, kMuted,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    const double total = g_position.finalScore(g_aValue);
    drawTextAt(dc, signedDecimal(total), x + scale(76), y, width - scale(76), scale(22), g_fontBig,
               total > 0.0 ? kRedSoft : (total < 0.0 ? RGB(174, 181, 195) : kGreen),
               DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    y += scale(26);

    const RECT chart{x, y, x + width, box.bottom - scale(12)};
    drawChart(dc, chart);
}

// ---------------------------------------------------------------------------
// Right panel: the record tree
// ---------------------------------------------------------------------------
void drawRecordTree(HDC dc, const Layout& layout) {
    const RECT& box = layout.right;
    roundRect(dc, box, scale(12), kPanel);
    drawTextAt(dc, L"棋谱", box.left + scale(12), box.top + scale(8), scale(60), scale(20),
               g_fontSmall, kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    const std::wstring count = std::to_wstring(g_record.depth(g_record.current())) + L" / " +
                               std::to_wstring(g_record.count() - 1);
    drawTextAt(dc, count, box.right - scale(60), box.top + scale(8), scale(48), scale(20),
               g_fontSmall, kMuted, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

    const TreeLayout tree = computeTreeLayout(layout, g_treeScroll);
    // Clip: a long game must not paint over the panel's border.
    HRGN clip = CreateRectRgn(box.left + 1, box.top + scale(30), box.right - 1, box.bottom - 1);
    SelectClipRgn(dc, clip);

    const std::vector<int> path = g_record.line(g_record.current());
    const auto onPath = [&](int id) { return std::find(path.begin(), path.end(), id) != path.end(); };
    for (const TreeCell& cell : tree.cells) {
        if (cell.box.bottom < box.top + scale(30)) continue;
        if (cell.box.top > box.bottom) break;
        const RecordNode& node = g_record.node(cell.node);
        const bool selected = cell.node == g_record.current();
        const bool trunk = onPath(cell.node);
        const int height = cell.box.bottom - cell.box.top;
        if (cell.variation) {
            // A variation is a note in the margin: a tick and a dimmer row.
            line(dc, cell.box.left - scale(7), cell.box.top, cell.box.left - scale(7),
                 cell.box.top + height / 2, RGB(96, 106, 124));
            line(dc, cell.box.left - scale(7), cell.box.top + height / 2, cell.box.left - scale(2),
                 cell.box.top + height / 2, RGB(96, 106, 124));
        }
        if (selected) {
            roundRect(dc, cell.box, scale(3), kPanelSoft, kGold);
        } else if (trunk) {
            roundRect(dc, cell.box, scale(3), kPanelDeep);
        }
        // What the move did to the number the position before it gave: a small
        // block on the left of the row, never the row's own colour.  A fill reads
        // as "this row is tagged", a block reads as "this move scored x", and it
        // stays visible under the selection, which is exactly when you are looking
        // at the move in question.
        MoveVerdict verdict;
        const bool hasVerdict =
            node.parent >= 0 && moveVerdict(g_record.node(node.parent), node, &verdict);
        if (hasVerdict) {
            // A significant rise on a move the engine never scored is the one case
            // worth a different colour: it evaluated the position, not the move,
            // and the move beat that verdict.  Everything else uses the board's own
            // scale -- green at 0, red at -10 points.
            const bool found = !verdict.scored && verdict.delta > kRowGainWindow;
            const COLORREF tint = found ? kCyan : markColor(verdict.delta, false);
            const int inset = scale(1);
            RECT dash{cell.box.left + scale(1), cell.box.top + inset,
                      cell.box.left + scale(5), cell.box.bottom - inset};
            roundRect(dc, dash, scale(1), tint);
        }
        const int textLeft = cell.box.left + scale(6);
        if (cell.node != g_record.root()) {
            // The ply number, so a row tells you where in the game it is.
            const std::wstring ply = std::to_wstring(g_record.depth(cell.node));
            drawTextAt(dc, ply, textLeft, cell.box.top, scale(24), height, g_fontSmall,
                       cell.variation ? RGB(120, 126, 138) : kMuted,
                       DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
        }
        const std::wstring label =
            cell.node == g_record.root() ? std::wstring(L"起") : moveLabel(node.move);
        drawTextAt(dc, label, textLeft + scale(28), cell.box.top, cell.box.right - textLeft - scale(30),
                   height, g_fontSmall,
                   selected ? kText : (trunk ? RGB(226, 231, 240) : kMuted),
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        if (node.analysis.valid && node.analysis.proven) {
            filledCircle(dc, cell.box.right - scale(8), cell.box.top + height / 2, scale(2), kGold);
        }
    }
    SelectClipRgn(dc, nullptr);
    DeleteObject(clip);
}

// ---------------------------------------------------------------------------
// Bottom bar
// ---------------------------------------------------------------------------
void drawBottomBar(HDC dc, const Layout& layout, const Chrome& chrome) {
    fillRectColor(dc, layout.bottom, kPanel);
    line(dc, 0, layout.bottom.top, layout.bottom.right, layout.bottom.top, RGB(48, 54, 65));

    const int barTop = layout.bottom.top;
    // A value with its two buttons.  It is a number the game is scored by, so it
    // stays visible, but it is not a headline.
    const int aRow = barTop + scale(21);
    drawTextAt(dc, L"A值", scale(16), aRow, scale(34), scale(21), g_fontSmall, kMuted,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, signedValue(g_aValue), scale(50), aRow, scale(44), scale(21), g_fontUi, kGold,
               DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    roundRect(dc, chrome.aMinus, scale(4), kPanelSoft, RGB(68, 75, 88));
    roundRect(dc, chrome.aPlus, scale(4), kPanelSoft, RGB(68, 75, 88));
    drawTextAt(dc, L"−", chrome.aMinus.left, chrome.aMinus.top,
               chrome.aMinus.right - chrome.aMinus.left, chrome.aMinus.bottom - chrome.aMinus.top,
               g_fontUi, kText);
    drawTextAt(dc, L"+", chrome.aPlus.left, chrome.aPlus.top, chrome.aPlus.right - chrome.aPlus.left,
               chrome.aPlus.bottom - chrome.aPlus.top, g_fontUi, kText);

    for (int side = 0; side < 2; ++side) {
        const Player player = side == 0 ? Player::Red : Player::Black;
        const int rowY = chrome.chips[side][0].top;
        drawTextAt(dc, side == 0 ? L"红" : L"黑", scale(168), rowY, scale(18), scale(21),
                   g_fontSmall, side == 0 ? kRedSoft : kMuted,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        const bool active = g_position.sideToMove() == player && humanToMove();
        for (int n = 1; n <= 9; ++n) {
            RECT chip = chrome.chips[side][n - 1];
            const int remain = g_position.remaining(player, n);
            const bool selected = active && n == g_selected;
            roundRect(dc, chip, scale(4), remain > 0 ? kPanelSoft : kPanelDeep,
                      selected ? kGold : RGB(58, 64, 76), selected ? 2 : 1);
            // The number, then how many of it are left as a superscript: "9³" means
            // three nines still in hand.  Without the count the only signal was a
            // dimmed chip, which reads as "used" but not as "one left".
            const COLORREF numberColour = remain > 0
                                              ? (side == 0 ? RGB(240, 179, 163) : RGB(215, 220, 230))
                                              : RGB(94, 99, 109);
            const int chipWidth = chip.right - chip.left;
            const int chipHeight = chip.bottom - chip.top;
            // Draw the digit and its count as one group ("9³"), not as two columns:
            // the digit right-aligned to 58% of the chip, the count left-aligned
            // right after it and raised, which is what a superscript is.
            const int digitRight = chip.left + (chipWidth * 62) / 100;
            drawTextAt(dc, std::to_wstring(n), chip.left, chip.top, digitRight - chip.left,
                       chipHeight, g_fontSmall, numberColour,
                       DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
            drawTextAt(dc, std::to_wstring(remain), digitRight, chip.top + scale(2),
                       scale(12), scale(11), g_fontTiny,
                       remain > 0 ? mixColor(numberColour, kPanelSoft, 0.15) : RGB(94, 99, 109),
                       DT_LEFT | DT_TOP | DT_SINGLELINE);
        }
    }

    const std::wstring ply = L"第 " + std::to_wstring(g_position.ply()) + L" 手";
    drawTextAt(dc, ply, layout.bottom.right - scale(120), barTop + scale(11), scale(100), scale(20),
               g_fontSmall, kMuted, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
}

void drawOverlay(HDC dc, const Layout& layout) {
    if (!g_position.gameOver()) return;
    const int boardSize = layout.board.right - layout.board.left;
    const int width = std::min(scale(360), boardSize - scale(30));
    const int height = scale(96);
    const int cx = layout.board.left + boardSize / 2;
    const int cy = layout.board.top + boardSize / 2;
    const RECT box{cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2};
    roundRect(dc, box, scale(14), RGB(31, 36, 45), kGold, scale(2));
    const Outcome outcome = g_position.outcome(g_aValue);
    const COLORREF colour = outcome == Outcome::RedWin ? RGB(240, 137, 122)
                             : outcome == Outcome::BlackWin ? RGB(196, 202, 214)
                                                           : kGreen;
    drawTextAt(dc, outcomeText(outcome), box.left + scale(16), box.top + scale(12),
               width - scale(32), scale(46), g_fontTitle, colour);
    const std::wstring detail = L"局面分 " + signedText(g_position.rawScore()) + L"　+　A值 " +
                                signedValue(g_aValue) + L"　=　" + signedDecimal(g_position.finalScore(g_aValue));
    drawTextAt(dc, detail, box.left + scale(12), box.top + scale(58), width - scale(24), scale(24),
               g_fontSmall, kText);
}

void paintAll(HDC dc, const RECT& client) {
    fillRectColor(dc, client, kBackground);
    Layout layout = computeLayout(g_hwnd);
    const Chrome chrome = computeChrome(layout);
    const auto& analysis = currentAnalysis();
    // A position that carries an analysis shows it, engine running or not: a
    // record saved with numbers is the case this rule is for.
    const auto rankedCells = analysis.valid ? rankAnalysisCells(analysis, g_position)
                                            : std::vector<int>{};
    drawBoard(dc, layout, rankedCells);
    drawLeftPanel(dc, layout, rankedCells);
    drawRecordTree(dc, layout);
    drawBottomBar(dc, layout, chrome);
    drawOverlay(dc, layout);
}

void refresh(bool immediate) {
    InvalidateRect(g_hwnd, nullptr, FALSE);
    if (immediate) UpdateWindow(g_hwnd);
}

// ---------------------------------------------------------------------------
// Game actions
// ---------------------------------------------------------------------------
void syncModeChecks() {
    if (!g_menu) return;
    int id = ID_MODE_HUMAN_RED;
    if (g_mode == Mode::Pvp) id = ID_MODE_PVP;
    else if (g_mode == Mode::HumanBlack) id = ID_MODE_HUMAN_BLACK;
    CheckMenuRadioItem(g_modeMenu, ID_MODE_PVP, ID_MODE_HUMAN_BLACK, id, MF_BYCOMMAND);
}

void resetGame() {
    g_record = GameRecord();
    clearAnalysisState();
    g_treeScroll = 0;
    g_selected = 1;
    g_hoverCell = -1;
    g_thinking = false;
    g_enginePending = false;
    rebuildPosition();
    syncModeChecks();
    if (g_engineChoice != EngineChoice::Off && g_engineReady) syncEnginePosition();
    if (aiToMove()) maybeAiMove();
    else refresh(true);
}

void setMode(Mode mode) {
    g_mode = mode;
    resetGame();
}

void setAValue(double value) {
    const double adjusted = normalizeAValue(value);
    if (adjusted == g_aValue) return;
    g_aValue = adjusted;
    saveSettings();
    if (g_engineChoice != EngineChoice::Off && g_engineReady) {
        sendEngine("set_a " + formatAValue(g_aValue));
        syncEnginePosition();
    }
    refresh();
}

void selectNumber(int number) {
    if (!humanToMove() || g_thinking) return;
    if (number < 1 || number > 9) return;
    if (g_position.remaining(g_position.sideToMove(), number) <= 0) return;
    g_selected = number;
    refresh();
}

void needEngine(const wchar_t* detail) {
    g_thinking = false;
    g_engineMessage = detail ? detail : L"没有可用的引擎：引擎菜单 → 选择引擎程序…";
    refresh();
}

// One move, because the user asked for it.  Opening a record or stepping through
// the tree deliberately does not do this: a move the user did not ask for lands in
// the record as a new branch, and from then on the game looks like a variation.
void computerMove() {
    if (!aiToMove()) {
        MessageBeep(MB_OK);
        return;
    }
    maybeAiMove();
}

void maybeAiMove() {
    if (!aiToMove() || g_thinking || g_enginePending) return;
    if (g_engineChoice != EngineChoice::Off && g_engineReady && g_engineClient.running()) {
        g_enginePending = true;
        g_thinking = true;
        // The genmove reply streams its own `info` lines, and they are the analysis
        // of this very position: keep them on this node.  `g_analysisOn` is left
        // alone so an analysis that was running resumes after the move.
        g_analysisNode = g_record.current();
        g_engineMessage = L"引擎思考中…";
        sendEngine("stop");
        g_pendingReplyId = std::to_string(++g_commandId);
        sendEngine(g_pendingReplyId + " genmove " +
                   std::string(g_position.sideToMove() == Player::Red ? "red" : "black") + " " +
                   std::to_string(std::max(1, g_playVisits)));
        refresh(true);
        return;
    }
    needEngine(L"没有可用的引擎：引擎菜单 → 选择引擎程序…");
}

void handleBoardClick(int cell) {
    if (g_thinking || !humanToMove() || g_position.gameOver()) return;
    if (g_position.stoneAt(cell) != 0) {
        MessageBeep(MB_OK);
        return;
    }
    if (g_position.remaining(g_position.sideToMove(), g_selected) <= 0) {
        g_selected = firstAvailable(g_position.sideToMove());
    }
    const Move move(cell, g_selected);
    const Player mover = g_position.sideToMove();
    if (!g_position.isLegal(move)) {
        MessageBeep(MB_ICONWARNING);
        return;
    }
    recordMove(move, mover);
    if (aiToMove()) maybeAiMove();
}

void handleClick(POINT point) {
    Layout layout = computeLayout(g_hwnd);
    Chrome chrome = computeChrome(layout);

    if (contains(chrome.aMinus, point)) {
        setAValue(g_aValue - kAValueStep);
        return;
    }
    if (contains(chrome.aPlus, point)) {
        setAValue(g_aValue + kAValueStep);
        return;
    }
    // The pieces in hand: clicking one selects that number.  It is the primary
    // way to pick one, so it stays on the board's controls, not in a panel.
    for (int side = 0; side < 2; ++side) {
        if (side != static_cast<int>(g_position.sideToMove())) continue;
        for (int n = 0; n < 9; ++n) {
            if (contains(chrome.chips[side][n], point)) {
                selectNumber(n + 1);
                return;
            }
        }
    }
    // The tree: click a node to go back (or forward) to it.
    if (contains(layout.right, point)) {
        const TreeLayout tree = computeTreeLayout(layout, g_treeScroll);
        for (const TreeCell& cell : tree.cells) {
            if (contains(cell.box, point)) {
                goToNode(cell.node);
                return;
            }
        }
        return;
    }
    if (contains(layout.board, point)) {
        const int x = (point.x - layout.board.left) / layout.cell;
        const int row = (point.y - layout.board.top) / layout.cell;
        if (x >= 0 && x < 9 && row >= 0 && row < 9) handleBoardClick(index(x, 8 - row));
    }
}

void updateHover(POINT point) {
    Layout layout = computeLayout(g_hwnd);
    int next = -1;
    if (contains(layout.board, point)) {
        const int x = (point.x - layout.board.left) / layout.cell;
        const int row = (point.y - layout.board.top) / layout.cell;
        if (x >= 0 && x < 9 && row >= 0 && row < 9) next = index(x, 8 - row);
    }
    if (next != g_hoverCell) {
        g_hoverCell = next;
        refresh();
    }
}

// ---------------------------------------------------------------------------
// Record files
// ---------------------------------------------------------------------------
bool saveRecordToFile() {
    wchar_t file[MAX_PATH] = L"shuyi-record.syp";
    std::wstring filter;
    filter += L"数弈棋谱 (*.syp)";
    filter.push_back(0);
    filter += L"*.syp";
    filter.push_back(0);
    filter += L"所有文件 (*.*)";
    filter.push_back(0);
    filter += L"*.*";
    filter.push_back(0);
    filter.push_back(0);
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = g_hwnd;
    dialog.lpstrFilter = filter.c_str();
    dialog.lpstrFile = file;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrDefExt = L"syp";
    dialog.lpstrTitle = L"保存棋谱";
    dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameW(&dialog)) return false;

    const std::string text = g_record.save(g_ruleset);
    std::FILE* handle = shuyi::fopenUtf8(narrowUtf8(file), "wb");
    if (!handle) {
        MessageBoxW(g_hwnd, L"无法写入这个文件。", L"保存棋谱", MB_OK | MB_ICONERROR);
        return false;
    }
    std::fwrite(text.data(), 1, text.size(), handle);
    std::fclose(handle);
    g_engineMessage = L"棋谱已保存";
    refresh();
    return true;
}

// Reads a record file into the window.  Shared by the dialog and `--open`, so a
// .syp file can be opened by double-clicking it.
bool loadRecordFromPath(const std::wstring& path) {
    std::FILE* handle = shuyi::fopenUtf8(narrowUtf8(path), "rb");
    if (!handle) {
        MessageBoxW(g_hwnd, (L"无法打开这个文件： " + path).c_str(), L"载入棋谱",
                    MB_OK | MB_ICONERROR);
        return false;
    }
    std::string text;
    char buffer[4096];
    for (std::size_t got = std::fread(buffer, 1, sizeof(buffer), handle); got > 0;
         got = std::fread(buffer, 1, sizeof(buffer), handle)) {
        text.append(buffer, got);
    }
    std::fclose(handle);
    return loadRecordFromText(text);
}

bool loadRecordFromFile() {
    wchar_t file[MAX_PATH] = L"";
    std::wstring filter;
    filter += L"数弈棋谱 (*.syp)";
    filter.push_back(0);
    filter += L"*.syp";
    filter.push_back(0);
    filter += L"所有文件 (*.*)";
    filter.push_back(0);
    filter += L"*.*";
    filter.push_back(0);
    filter.push_back(0);
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = g_hwnd;
    dialog.lpstrFilter = filter.c_str();
    dialog.lpstrFile = file;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrDefExt = L"syp";
    dialog.lpstrTitle = L"载入棋谱";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetOpenFileNameW(&dialog)) return false;

    return loadRecordFromPath(file);
}

bool loadRecordFromText(const std::string& text) {
    const Ruleset previousRuleset = g_ruleset;
    GameRecord loaded;
    Ruleset ruleset = g_ruleset;
    std::string error;
    if (!loaded.load(text, &ruleset, &error)) {
        MessageBoxW(g_hwnd, (L"这个文件不是数弈棋谱：\n" + widen(error)).c_str(), L"载入棋谱",
                    MB_OK | MB_ICONERROR);
        return false;
    }
    g_record = std::move(loaded);
    g_ruleset = ruleset;
    clearAnalysisState();
    g_engineMessage = L"棋谱已载入";
    rebuildPosition();
    syncRulesetChecks();
    saveSettings();
    if (g_engineChoice != EngineChoice::Off && g_engineReady) {
        // The engine takes the rule set at start-up, so only a record with another
        // rule set needs a restart; otherwise replaying the line is enough.  The
        // restart is also the fragile path (a probe racing the process it just
        // killed), so it is worth not taking it when nothing changed.
        if (ruleset != previousRuleset) startEngineProcess();
        else syncEnginePosition();
    }
    g_treeScroll = std::max(0, treeScrollFor(g_record.current()));
    noteComputerToMove();
    refresh(true);
    return true;
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------
void openProjectPage() {
    ShellExecuteW(g_hwnd, L"open", kProjectUrl, nullptr, nullptr, SW_SHOWNORMAL);
}

void showGuide() {
    std::wstring text;
    text += L"数弈 · 规则\n\n";
    text += L"1. 9×9 棋盘，横坐标 a–i，纵坐标 1–9。\n";
    text += L"2. 红黑双方，红先，轮流在空格落子；棋子落下后不可移动、不可移除。\n";
    text += L"3. 每方有数字 1–9 各三枚，共 27 枚。同数字棋子彼此完全相同。\n";
    text += L"4. 双方 54 枚棋子全部落完后终局。\n\n";
    text += L"计分：单格分 = sgn(附近红子最大数字 − 附近黑子最大数字)，"
            L"“附近”指曼哈顿距离 ≤1 的格子（含该格自身）。\n";
    text += L"局面分 = 81 个格子分之和 + A值。局面分 >0 红胜，<0 黑胜，=0 平局。\n\n";
    text += L"操作\n\n";
    text += L"· 数字键 1–9：选择落子的数字（也可以点下方棋子）。\n";
    text += L"· 点击棋盘：落子。\n";
    text += L"· 空格：开始 / 暂停分析；分析到上限后再按一次会继续，不再限量。开着分析落子，"
            L"会在新局面自动接着算（到过上限也一样）。\n";
    text += L"· ↑ / ↓：在棋谱里上一步 / 下一步（↓ 走主干）。翻棋谱不会让电脑落子，"
            L"要它走一步用 引擎 → 让电脑走一步。\n";
    text += L"· Delete：删除当前着法及其后续，连同它们的分析。\n";
    text += L"· Ctrl+S：保存棋谱（等同 文件 → 保存棋谱…）。\n";
    text += L"· 左侧面板是分析的产物，右侧是棋谱：每一着一行，换个走法会在那一手后面缩进出一段"
            L"变着，回到旧行时原来的分析还在。\n";
    text += L"· 棋盘上每个分析标记：胜率 / 访问数 / 预期终局分，右上角的上标是要落的那颗子的数字"
            L"（同一个格子能下九个数字）。\n";
    text += L"· 棋谱每行左边的小色块是这一步相对上一步的评估：蓝 = 比之前明显更好，"
            L"绿 = 不差，红 = 变差。\n";
    text += L"· 分析会一起存进棋谱；载入棋谱时，每个局面的分析跟着回来。\n";
    text += L"· 下方每一步棋子右上角的小数字是这种数字还剩几颗（9³ = 还有三颗 9）。\n";
    text += L"· 左下角 A值 可以按 ± 调整（也可在 引擎 → 设置 里改）。\n";
    text += L"· 新局、保存棋谱、载入棋谱在 文件 菜单里。\n\n";
    text += L"项目主页（规则、引擎、模型的发布都在这里）：\n";
    text += kProjectUrl;
    MessageBoxW(g_hwnd, text.c_str(), L"使用指南", MB_OK | MB_ICONINFORMATION);
}

void showAbout() {
    std::wstring text;
    text += L"数弈\n\n";
    text += L"一个 9×9 的数字落子棋。原生 Win32/GDI 界面，非 Web 实现。\n";
    text += L"界面只负责规则、局面和显示；搜索与计算全部在这个窗口之外的引擎进程里，"
            L"两者只通过行协议对话。\n\n";
    text += L"引擎模型与源码的发布：\n";
    text += kProjectUrl;
    MessageBoxW(g_hwnd, text.c_str(), L"关于数弈", MB_OK | MB_ICONINFORMATION);
}

// ---------------------------------------------------------------------------
// Fonts / menus / window
// ---------------------------------------------------------------------------
HFONT makeFont(int points, bool bold, const wchar_t* face = L"Microsoft YaHei UI") {
    LOGFONTW lf{};
    lf.lfHeight = -MulDiv(points, g_dpi, 72);
    lf.lfWeight = bold ? FW_BOLD : FW_NORMAL;
    lf.lfCharSet = DEFAULT_CHARSET;
    lf.lfQuality = ANTIALIASED_QUALITY;
    lstrcpynW(lf.lfFaceName, face, LF_FACESIZE);
    return CreateFontIndirectW(&lf);
}

void createFonts() {
    g_fontTitle = makeFont(24, true);
    g_fontSubtitle = makeFont(10, false);
    g_fontTurn = makeFont(15, true);
    g_fontBig = makeFont(17, true);
    g_fontPiece = makeFont(19, true);
    g_fontUi = makeFont(12, false);
    g_fontSmall = makeFont(10, false);
    g_fontCoord = makeFont(10, false);
    g_fontTiny = makeFont(8, false);
}

void destroyFonts() {
    for (HFONT* font : {&g_fontTitle, &g_fontSubtitle, &g_fontTurn, &g_fontBig, &g_fontPiece,
                        &g_fontUi, &g_fontSmall, &g_fontCoord, &g_fontTiny}) {
        if (*font) DeleteObject(*font);
        *font = nullptr;
    }
}

void createMenuBar(HWND hwnd) {
    HMENU menu = CreateMenu();
    HMENU file = CreatePopupMenu();
    HMENU mode = CreatePopupMenu();
    HMENU engine = CreatePopupMenu();
    HMENU rulesetMenu = CreatePopupMenu();
    HMENU help = CreatePopupMenu();
    HMENU model = CreatePopupMenu();

    AppendMenuW(file, MF_STRING, ID_NEW_GAME, L"新局");
    AppendMenuW(file, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(file, MF_STRING, ID_SAVE_RECORD, L"保存棋谱…\tCtrl+S");
    AppendMenuW(file, MF_STRING, ID_LOAD_RECORD, L"载入棋谱…");
    AppendMenuW(file, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(file, MF_STRING, ID_EXIT, L"退出");

    AppendMenuW(rulesetMenu, MF_STRING, ID_RULESET_NOPASS,
                L"无过手（阻挡规则；一方无子可落立刻终局）");
    AppendMenuW(rulesetMenu, MF_STRING, ID_RULESET_BLOCKING,
                L"阻挡规则（异色大子禁止相邻；无子可落时强制过手）");
    AppendMenuW(rulesetMenu, MF_STRING, ID_RULESET_CLASSIC, L"现行规则（无阻挡）");

    AppendMenuW(mode, MF_STRING, ID_MODE_PVP, L"双人对弈");
    AppendMenuW(mode, MF_STRING, ID_MODE_HUMAN_RED, L"人机 · 你执红");
    AppendMenuW(mode, MF_STRING, ID_MODE_HUMAN_BLACK, L"人机 · 你执黑");

    AppendMenuW(engine, MF_STRING, ID_ENGINE_OFF, L"不使用外部引擎");
    AppendMenuW(engine, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(engine, MF_STRING, ID_ENGINE_MCTS, L"MCTS（无网络）");
    // One entry per discovered model, filled in by populateModelMenu().
    AppendMenuW(engine, MF_POPUP | MF_STRING, reinterpret_cast<UINT_PTR>(model), L"MCTS + 网络(&N)");
    g_modelMenuPosition = GetMenuItemCount(engine) - 1;
    AppendMenuW(engine, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(engine, MF_STRING, ID_ENGINE_MOVE, L"让电脑走一步");
    AppendMenuW(engine, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(engine, MF_STRING, ID_ENGINE_PICK, L"选择引擎程序…");
    AppendMenuW(engine, MF_STRING, ID_ENGINE_AUTO, L"自动探测引擎");
    AppendMenuW(engine, MF_STRING, ID_ENGINE_INFO, L"引擎信息…");
    AppendMenuW(engine, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(engine, MF_STRING, ID_SETTINGS, L"设置（搜索量 / A值）…");

    AppendMenuW(help, MF_STRING, ID_GUIDE, L"使用指南");
    AppendMenuW(help, MF_STRING, ID_PROJECT_PAGE, L"打开项目主页");
    AppendMenuW(help, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(help, MF_STRING, ID_ABOUT, L"关于数弈");

    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(file), L"文件(&F)");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(mode), L"模式(&M)");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(rulesetMenu), L"规则(&R)");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(engine), L"引擎(&E)");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(help), L"帮助(&H)");
    SetMenu(hwnd, menu);
    g_menu = menu;
    g_modeMenu = mode;
    g_rulesetMenu = rulesetMenu;
    g_engineMenu = engine;
    g_modelMenu = model;
    scanModels();
    populateModelMenu();
    syncModeChecks();
    syncEngineChecks();
    syncRulesetChecks();
}

LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE:
        g_hwnd = hwnd;
        g_dpi = queryDpi();
        createFonts();
        createMenuBar(hwnd);
        SetTimer(hwnd, 1, 100, nullptr);
        return 0;

    case WM_APP_RESET:
        resetGame();
        if (!g_pendingOpen.empty()) {
            const std::wstring path = g_pendingOpen;
            g_pendingOpen.clear();
            loadRecordFromPath(path);
        }
        return 0;

    case WM_TIMER:
        if (wParam == 1) {
            drainEngineLines();
            if (g_engineChoice != EngineChoice::Off && (g_analysisOn || g_enginePending)) refresh();
        }
        return 0;

    case WM_GETMINMAXINFO: {
        auto* info = reinterpret_cast<MINMAXINFO*>(lParam);
        info->ptMinTrackSize.x = scale(940);
        info->ptMinTrackSize.y = scale(680);
        return 0;
    }

    case WM_ERASEBKGND:
        return 1;

    case WM_PAINT: {
        PAINTSTRUCT paint;
        HDC dc = BeginPaint(hwnd, &paint);
        RECT client;
        GetClientRect(hwnd, &client);
        HDC memory = CreateCompatibleDC(dc);
        HBITMAP bitmap = CreateCompatibleBitmap(dc, client.right, client.bottom);
        HGDIOBJ old = SelectObject(memory, bitmap);
        paintAll(memory, client);
        BitBlt(dc, 0, 0, client.right, client.bottom, memory, 0, 0, SRCCOPY);
        SelectObject(memory, old);
        DeleteObject(bitmap);
        DeleteDC(memory);
        EndPaint(hwnd, &paint);
        return 0;
    }

    case WM_LBUTTONDOWN: {
        SetFocus(hwnd);
        POINT point{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
        handleClick(point);
        return 0;
    }

    case WM_MOUSEMOVE: {
        POINT point{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
        updateHover(point);
        TRACKMOUSEEVENT track{sizeof(TRACKMOUSEEVENT), TME_LEAVE, hwnd, 0};
        TrackMouseEvent(&track);
        return 0;
    }

    case WM_MOUSELEAVE:
        if (g_hoverCell != -1) {
            g_hoverCell = -1;
            refresh();
        }
        return 0;

    case WM_MOUSEWHEEL: {
        // Wheel messages carry screen coordinates, unlike the mouse messages above.
        POINT point{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
        ScreenToClient(hwnd, &point);
        Layout layout = computeLayout(hwnd);
        if (contains(layout.right, point)) {
            const TreeLayout tree = computeTreeLayout(layout, 0);
            const int visible = layout.right.bottom - (layout.right.top + scale(34));
            const int maximum = std::max(0, tree.contentHeight - visible + scale(20));
            const int delta = GET_WHEEL_DELTA_WPARAM(wParam) / WHEEL_DELTA * scale(38);
            g_treeScroll = std::clamp(g_treeScroll - delta, 0, maximum);
            refresh();
        }
        return 0;
    }

    case WM_KEYDOWN: {
        switch (wParam) {
        case VK_SPACE:
            toggleAnalysis();
            return 0;
        case VK_UP: {
            const int id = g_record.parentOf(g_record.current());
            if (id >= 0) {
                goToNode(id);
                g_treeScroll = treeScrollFor(g_record.current());
            }
            return 0;
        }
        case VK_DOWN: {
            const auto& children = g_record.childrenOf(g_record.current());
            if (!children.empty()) {
                goToNode(children.front());
                g_treeScroll = treeScrollFor(g_record.current());
            }
            return 0;
        }
        case VK_DELETE:
            if (g_record.current() != g_record.root()) {
                // Erasing remaps every node id, so nothing may hold one across it:
                // stop the analysis and forget which node it belonged to first.
                sendEngineStopIfRunning();
                clearAnalysisState();
                g_record.erase(g_record.current());
                rebuildPosition();
                g_treeScroll = treeScrollFor(g_record.current());
                if (g_engineChoice != EngineChoice::Off && g_engineReady) syncEnginePosition();
                refresh(true);
                if (aiToMove()) maybeAiMove();
            }
            return 0;
        default:
            break;
        }
        if (wParam >= '1' && wParam <= '9') {
            selectNumber(static_cast<int>(wParam - '0'));
            return 0;
        }
        if (wParam >= VK_NUMPAD1 && wParam <= VK_NUMPAD9) {
            selectNumber(static_cast<int>(wParam - VK_NUMPAD0));
            return 0;
        }
        return 0;
    }

    case WM_COMMAND:
        if (LOWORD(wParam) >= static_cast<UINT>(ID_MODEL_BASE) &&
            LOWORD(wParam) < static_cast<UINT>(ID_MODEL_LIMIT)) {
            selectModel(static_cast<int>(LOWORD(wParam)) - ID_MODEL_BASE);
            return 0;
        }
        switch (LOWORD(wParam)) {
        case ID_NEW_GAME: resetGame(); return 0;
        case ID_EXIT: DestroyWindow(hwnd); return 0;
        case ID_SAVE_RECORD: saveRecordToFile(); return 0;
        case ID_LOAD_RECORD: loadRecordFromFile(); return 0;
        case ID_MODE_PVP: setMode(Mode::Pvp); return 0;
        case ID_MODE_HUMAN_RED: setMode(Mode::HumanRed); return 0;
        case ID_MODE_HUMAN_BLACK: setMode(Mode::HumanBlack); return 0;
        case ID_RULESET_CLASSIC: setRuleset(Ruleset::Classic); return 0;
        case ID_RULESET_BLOCKING: setRuleset(Ruleset::Blocking); return 0;
        case ID_RULESET_NOPASS: setRuleset(Ruleset::NoPass); return 0;
        case ID_ENGINE_OFF: setEngineChoice(EngineChoice::Off); return 0;
        case ID_ENGINE_MCTS: setEngineChoice(EngineChoice::Mcts); return 0;
        case ID_ENGINE_MCTS_NN: setEngineChoice(EngineChoice::MctsNn); return 0;
        case ID_ENGINE_PICK: pickEngineProgram(); return 0;
        case ID_ENGINE_AUTO: useAutoEngine(); return 0;
        case ID_ENGINE_MOVE: computerMove(); return 0;
        case ID_ENGINE_INFO: showEngineInfo(); return 0;
        case ID_MODEL_RESCAN: {
            scanModels();
            const int index = modelIndexFor(g_nnModelPath);
            if (index < 0) g_nnModelPath.clear();
            populateModelMenu();
            syncModelChecks();
            refresh();
            return 0;
        }
        case ID_SETTINGS: openSettings(); return 0;
        case ID_A_PLUS: setAValue(g_aValue + kAValueStep); return 0;
        case ID_A_MINUS: setAValue(g_aValue - kAValueStep); return 0;
        case ID_GUIDE: showGuide(); return 0;
        case ID_PROJECT_PAGE: openProjectPage(); return 0;
        case ID_ABOUT: showAbout(); return 0;
        default: break;
        }
        break;

    case WM_DESTROY:
        KillTimer(hwnd, 1);
        g_engineClient.stop();
        destroyFonts();
        PostQuitMessage(0);
        return 0;

    default:
        break;
    }
    return DefWindowProcW(hwnd, message, wParam, lParam);
}

} // namespace

// Headless check of the model menu: `shuyi.exe --dump-models out.txt` writes the
// exact entry list the MCTS+NN submenu builds, so the scan can be tested without
// a screenshot or a GUI session.
bool dumpEngineChoice(const std::string& path) {
    loadSettings();
    std::FILE* file = fopenUtf8(path, "w");
    if (!file) return false;
    char dirUtf8[MAX_PATH * 2] = "";
    WideCharToMultiByte(CP_UTF8, 0, engineClientDirectory().c_str(), -1, dirUtf8, sizeof(dirUtf8),
                        nullptr, nullptr);
    std::fprintf(file, "exeDir=%s", dirUtf8);
    std::fputc(10, file);
    {
        char probePath[MAX_PATH * 2] = "";
        const std::wstring guess = engineClientDirectory() + L"/shuyi_engine.exe";
        WideCharToMultiByte(CP_UTF8, 0, guess.c_str(), -1, probePath, sizeof(probePath), nullptr,
                            nullptr);
        std::fprintf(file, "guess=%s exists=%d", probePath,
                     GetFileAttributesW(guess.c_str()) == INVALID_FILE_ATTRIBUTES ? 0 : 1);
        std::fputc(10, file);
    }
    std::vector<std::wstring> candidates = engineCandidates();
    for (const std::wstring& tried : enginePathsTried()) {
        char utf8[MAX_PATH * 2] = "";
        WideCharToMultiByte(CP_UTF8, 0, tried.c_str(), -1, utf8, sizeof(utf8), nullptr, nullptr);
        std::fprintf(file, "try %d %s",
                     GetFileAttributesW(tried.c_str()) == INVALID_FILE_ATTRIBUTES ? 0 : 1, utf8);
        std::fputc(10, file);
    }
    std::fprintf(file, "candidates=%zu", candidates.size());
    std::fputc(10, file);
    for (const std::wstring& candidate : candidates) {
        const EngineProbe probe = probeEngine(candidate);
        char utf8[MAX_PATH * 2] = "";
        WideCharToMultiByte(CP_UTF8, 0, candidate.c_str(), -1, utf8, sizeof(utf8), nullptr, nullptr);
        std::fprintf(file, "%s    %s", probe.answered ? probe.backend.c_str() : "no-reply", utf8);
        std::fputc(10, file);
    }
    const std::wstring setting = g_enginePath.empty() ? std::wstring(L"(auto)") : g_enginePath;
    {
        char utf8[MAX_PATH * 2] = "";
        WideCharToMultiByte(CP_UTF8, 0, setting.c_str(), -1, utf8, sizeof(utf8), nullptr, nullptr);
        std::fprintf(file, "setting=%s", utf8);
        std::fputc(10, file);
    }
    const std::wstring chosen = g_enginePath.empty() ? chooseEnginePath() : g_enginePath;
    if (!chosen.empty()) {
        char utf8[MAX_PATH * 2] = "";
        WideCharToMultiByte(CP_UTF8, 0, chosen.c_str(), -1, utf8, sizeof(utf8), nullptr, nullptr);
        std::fprintf(file, "chosen=%s", utf8);
        std::fputc(10, file);
    }
    std::fclose(file);
    return true;
}

bool dumpModelMenu(const std::string& path) {
    g_modelMenu = CreatePopupMenu();
    scanModels();
    populateModelMenu();
    std::FILE* file = shuyi::fopenUtf8(path, "w");
    if (!file) return false;
    const int count = static_cast<int>(GetMenuItemCount(g_modelMenu));
    std::fprintf(file, "models=%d\n", static_cast<int>(g_modelList.size()));
    for (int position = 0; position < count; ++position) {
        wchar_t label[512] = L"";
        GetMenuStringW(g_modelMenu, static_cast<UINT>(position), label, 512, MF_BYPOSITION);
        const UINT id = GetMenuItemID(g_modelMenu, position);
        // The labels are Chinese, so print UTF-8 explicitly: %ls goes through the
        // C locale and stops at the first character it cannot represent.
        char utf8[2048] = "";
        WideCharToMultiByte(CP_UTF8, 0, label, -1, utf8, sizeof(utf8), nullptr, nullptr);
        std::fprintf(file, "%u\t%s\n", id, utf8);
    }
    std::fclose(file);
    DestroyMenu(g_modelMenu);
    g_modelMenu = nullptr;
    return true;
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR commandLine, int showCommand) {
    g_instance = instance;
    SetProcessDPIAware();
    g_dpi = queryDpi();
    loadSettings();
    // The position starts from the persisted rule set instead of the compiled-in
    // default, so a package can ship "blocking" by shipping its shuyi-ui.ini.
    g_position.setRuleset(g_ruleset);

    if (commandLine && std::wstring(commandLine).find(L"--dump-engine") != std::wstring::npos) {
        std::wstring argument = commandLine;
        const std::wstring flag = L"--dump-engine";
        const std::size_t at = argument.find(flag);
        std::wstring tail = argument.substr(at + flag.size());
        while (!tail.empty() && (tail.front() == L' ' || tail.front() == L'\t')) tail.erase(0, 1);
        std::string out = "engine_choice.txt";
        if (!tail.empty()) {
            const int needed =
                WideCharToMultiByte(CP_UTF8, 0, tail.c_str(), -1, nullptr, 0, nullptr, nullptr);
            if (needed > 1) {
                out.assign(static_cast<std::size_t>(needed - 1), '\0');
                WideCharToMultiByte(CP_UTF8, 0, tail.c_str(), -1, out.data(), needed, nullptr,
                                    nullptr);
            }
        }
        return dumpEngineChoice(out) ? 0 : 1;
    }
    if (commandLine && std::wstring(commandLine).find(L"--dump-models") != std::wstring::npos) {
        std::wstring argument = commandLine;
        const std::wstring flag = L"--dump-models";
        const std::size_t at = argument.find(flag);
        std::wstring tail = argument.substr(at + flag.size());
        while (!tail.empty() && (tail.front() == L' ' || tail.front() == L'\t')) tail.erase(0, 1);
        // The command line is UTF-16; the path may contain Chinese characters, so
        // convert it properly instead of narrowing code unit by code unit.
        std::string dumpPath = "ui_models.txt";
        if (!tail.empty()) {
            const int needed =
                WideCharToMultiByte(CP_UTF8, 0, tail.c_str(), -1, nullptr, 0, nullptr, nullptr);
            if (needed > 1) {
                dumpPath.assign(static_cast<std::size_t>(needed - 1), '\0');
                WideCharToMultiByte(CP_UTF8, 0, tail.c_str(), -1, dumpPath.data(), needed, nullptr,
                                    nullptr);
            }
        }
        const bool ok = dumpModelMenu(dumpPath);
        return ok ? 0 : 1;
    }

    // `shuyi.exe --open game.syp` loads a record; a .syp file association uses it.
    std::wstring openPath;
    if (commandLine) {
        const std::wstring argument(commandLine);
        const std::wstring flag = L"--open";
        const std::size_t at = argument.find(flag);
        if (at != std::wstring::npos) {
            std::wstring tail = argument.substr(at + flag.size());
            while (!tail.empty() && (tail.front() == L' ' || tail.front() == L'	')) tail.erase(0, 1);
            if (tail.size() > 1 && tail.front() == L'"' && tail.back() == L'"') {
                tail = tail.substr(1, tail.size() - 2);
            }
            openPath = tail;
        }
    }

    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = windowProc;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.hIconSm = LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.hbrBackground = nullptr;
    windowClass.lpszClassName = L"ShuyiNativeWindow";
    if (!RegisterClassExW(&windowClass)) return 1;

    // Second class for the settings popup: it must not run the board painting
    // code, so it has its own tiny window procedure.
    WNDCLASSEXW settingsClass = windowClass;
    settingsClass.lpfnWndProc = settingsProc;
    settingsClass.style = CS_HREDRAW | CS_VREDRAW;
    settingsClass.lpszClassName = kSettingsClass;
    if (!RegisterClassExW(&settingsClass)) return 1;

    RECT windowRect{0, 0, scale(1120), scale(800)};
    AdjustWindowRectEx(&windowRect, WS_OVERLAPPEDWINDOW, TRUE, 0);
    const int width = windowRect.right - windowRect.left;
    const int height = windowRect.bottom - windowRect.top;
    RECT workArea{};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &workArea, 0);
    const int x = workArea.left + ((workArea.right - workArea.left) - width) / 2;
    const int y = workArea.top + ((workArea.bottom - workArea.top) - height) / 2;

    g_hwnd = CreateWindowExW(0, windowClass.lpszClassName, L"数弈 — 数字落子棋",
                             WS_OVERLAPPEDWINDOW, x, y, width, height, nullptr, nullptr, instance,
                             nullptr);
    if (!g_hwnd) return 1;

    ShowWindow(g_hwnd, showCommand);
    UpdateWindow(g_hwnd);
    PostMessageW(g_hwnd, WM_APP_RESET, 0, 0);
    g_pendingOpen = openPath;

    // Ctrl+S saves the record.  It is the only key chord in the window: the menu
    // says so next to the command, which is where a user looks for it.
    ACCEL saveKeys[] = {{static_cast<BYTE>(FVIRTKEY | FCONTROL), 'S', ID_SAVE_RECORD}};
    HACCEL accelerators = CreateAcceleratorTableW(saveKeys, 1);

    // If an engine is discoverable next to the window, load it now so analysis is
    // one keypress away.  A package ships no weights, so "an engine but no model"
    // is a normal state: it must open quietly with the engine off, not greet the
    // user with an error dialog for a file they were never given.
    if (!chooseEnginePath().empty() && !defaultModelPath().empty()) {
        g_engineChoice = EngineChoice::MctsNn;
        if (!startEngineProcess()) {
            g_engineChoice = EngineChoice::Off;
            g_engineReady = false;
        }
        syncEngineChecks();
    } else if (!chooseEnginePath().empty()) {
        g_engineMessage = L"没有模型：把 .bin 放进 models/";
    }

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        if (accelerators && TranslateAcceleratorW(g_hwnd, accelerators, &message)) continue;
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    if (accelerators) DestroyAcceleratorTable(accelerators);
    return static_cast<int>(message.wParam);
}
