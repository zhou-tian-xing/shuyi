// 数弈 — native Win32/GDI user interface.  No web view, browser, or HTML.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <commdlg.h>

#include "shuyi/format.h"
#include "shuyi/position.h"
#include "shuyi/io.h"
#include "discovery.h"
#include "engine_client.h"
#include "engine_view.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace shuyi;

namespace {

// ---------------------------------------------------------------------------
// Command ids / app state
// ---------------------------------------------------------------------------
constexpr int ID_NEW = 1001;
constexpr int ID_UNDO = 1002;
constexpr int ID_EXIT = 1003;

constexpr int ID_MODE_PVP = 2001;
constexpr int ID_MODE_HUMAN_RED = 2002;
constexpr int ID_MODE_HUMAN_BLACK = 2003;

constexpr int ID_A_PLUS = 3001;
constexpr int ID_A_MINUS = 3002;
constexpr int ID_A_ZERO = 3003;
constexpr int ID_A_ONE = 3004;
constexpr int ID_A_TWO = 3005;

constexpr int ID_RULES = 4001;
constexpr int ID_ABOUT = 4002;

constexpr int ID_RULESET_CLASSIC = 5101;
constexpr int ID_RULESET_BLOCKING = 5102;
constexpr int ID_RULESET_NOPASS = 5103;
constexpr int ID_ENGINE_OFF = 5001;
constexpr int ID_ENGINE_PRUNING = 5002;
constexpr int ID_ENGINE_MCTS = 5003;
constexpr int ID_ENGINE_MCTS_NN = 5004;  // popup: one entry per model file
constexpr int ID_ENGINE_PICK = 5007;
constexpr int ID_ENGINE_AUTO = 5008;
constexpr int ID_ENGINE_INFO = 5009;
constexpr int ID_ANALYZE = 5005;
constexpr int ID_ENGINE_STOP = 5006;
constexpr int ID_SETTINGS = 5010;
constexpr int ID_MODEL_RESCAN = 6090;
constexpr int ID_MODEL_BASE = 6100;  // + model index
constexpr int ID_MODEL_LIMIT = 6400; // exclusive upper bound, below nothing else
constexpr int ID_SETTINGS_OK = 6001;
constexpr int ID_SETTINGS_CANCEL = 6002;

// Simulations per analysis and per move are settings (default 512).  A visit
// budget is the only machine-independent unit: 512 visits is 512 visits on the
// laptop and on the GPU, so two runs of the same position are comparable.  The
// default matches the evaluation budget in training/evalconfig.py.
constexpr int kDefaultVisits = 512;
// Window class of the settings popup (registered in wWinMain).
constexpr const wchar_t* kSettingsClass = L"ShuyiSettingsWindow";

constexpr UINT WM_APP_RESET = WM_APP + 1;

enum class Mode { Pvp, HumanRed, HumanBlack };
enum class EngineChoice { Off, Pruning, Mcts, MctsNn };

HWND g_hwnd = nullptr;
HMENU g_menu = nullptr;
HMENU g_modeMenu = nullptr;
HMENU g_rulesetMenu = nullptr;
HMENU g_engineMenu = nullptr;
HMENU g_modelMenu = nullptr;
int g_modelMenuPosition = -1;
// Every *.bin the UI can find, indexed to match menu ids ID_MODEL_BASE + i.
struct ModelEntry {
    // Win32 *A functions speak the ANSI code page, so paths found by scanning are
    // ANSI.  The engine process re-parses its command line through Utf8Args, so it
    // needs UTF-8.  Keep both instead of guessing at each use.
    std::string path;        // ANSI, for Win32 / fopen
    std::string enginePath;  // UTF-8, what set_model receives
    std::wstring label;      // "run-name/file.bin", unique in the menu
    int ruleset = 0;         // 0 classic, 1 blocking, -1 unknown
    int steps = 0;           // training steps from "-s<steps>-", 0 when unknown
};
std::vector<ModelEntry> g_modelList;
std::string g_nnModelPath;
// The engine is a separate program.  Its path is a user setting (shuyi-ui.ini),
// not something guessed from where this executable happens to live.
std::wstring g_enginePath;      // configured by the user, persisted
std::wstring g_enginePathUsed;  // what the running process actually is
bool g_engineAuto = true;       // no explicit path: fall back to discovery
HINSTANCE g_instance = nullptr;
Position g_position;
// nopass 的标定值是 +0.5（A* = 0.28~0.55，见 docs/RULES.zh-CN.md §4.4）：
// 半整数、不会和棋。INI 里的 a_value 会覆盖它。
double g_aValue = 0.5;
int g_selected = 1;
int g_hoverCell = -1;
bool g_thinking = false;
// Two players by default: the window is a board and an analysis tool first.
Mode g_mode = Mode::Pvp;
int g_dpi = 96;

EngineClient g_engineClient;
EngineChoice g_engineChoice = EngineChoice::Off;
EngineAnalysisView g_engineAnalysis;
bool g_analysisOn = false;
// The last analysis spent its whole budget: the next Space continues the same
// tree without a limit instead of starting over at the configured budget.
bool g_analysisFinished = false;
// Budgets, editable in the settings window and persisted in shuyi-ui.ini.  0
// means "no limit" (the search runs until Space is pressed again).
int g_analysisVisits = kDefaultVisits;
int g_playVisits = kDefaultVisits;
bool g_enginePending = false;
// Id of the `genmove` whose reply we are waiting for.  Without it a stray `= ` ack
// from `play`/`clear_board` (written before the genmove reply) would be parsed as
// a move.  The protocol allows an id on every command for exactly this reason.
std::string g_pendingReplyId;
int g_commandId = 0;
std::wstring g_engineMessage = L"外部引擎未启用";
bool g_engineReady = false;
// Rule set of the current game.  Switching it is only allowed before the first
// stone, because the two rules cannot mix inside one game.
Ruleset g_ruleset = kDefaultRuleset;

HFONT g_fontTitle = nullptr;
HFONT g_fontSubtitle = nullptr;
HFONT g_fontTurn = nullptr;
HFONT g_fontBig = nullptr;
HFONT g_fontPiece = nullptr;
HFONT g_fontUi = nullptr;
HFONT g_fontSmall = nullptr;
HFONT g_fontCoord = nullptr;

const COLORREF kBackground = RGB(22, 26, 33);
const COLORREF kPanel = RGB(36, 41, 50);
const COLORREF kPanelSoft = RGB(45, 51, 62);
const COLORREF kPanelDeep = RGB(29, 33, 41);
const COLORREF kBoard = RGB(225, 201, 153);
const COLORREF kBoardLight = RGB(240, 220, 178);
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
const COLORREF kBlue = RGB(64, 142, 255);
const COLORREF kHeatLow = RGB(185, 56, 52);
const COLORREF kHeatHigh = RGB(49, 157, 92);

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
void needEngine(const wchar_t* detail);
void syncEngineChecks();

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

// The A value lives on a half-integer grid (an integer A can draw, a half-integer
// cannot), so it prints without trailing zeros: "1.5", "1", "-0.5".
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
    // Never "+-0.0": an exact zero has no sign to show.
    if (value == 0.0) value = 0.0;
    std::wostringstream out;
    if (value >= 0.0) out << L'+';
    out << std::fixed << std::setprecision(1) << value;
    return out.str();
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

std::wstring turnLabel() {
    if (g_thinking) return L"电脑思考中…";
    if (g_position.gameOver()) {
        const Outcome outcome = g_position.outcome(g_aValue);
        return std::wstring(L"终局 · ") + outcomeText(outcome);
    }
    const Player side = g_position.sideToMove();
    std::wstring label = side == Player::Red ? L"红方行棋" : L"黑方行棋";
    if (g_mode != Mode::Pvp) {
        const bool human = (g_mode == Mode::HumanRed) ? side == Player::Red : side == Player::Black;
        label += human ? L" · 你" : L" · 电脑";
    }
    return label;
}

std::wstring modeLabel() {
    switch (g_mode) {
    case Mode::Pvp: return L"模式：双人对弈";
    case Mode::HumanRed: return L"模式：人机 · 你执红";
    case Mode::HumanBlack: return L"模式：人机 · 你执黑";
    }
    return L"";
}

// ---------------------------------------------------------------------------
// External engine client
// ---------------------------------------------------------------------------
const char* engineChoiceName(EngineChoice choice) {
    switch (choice) {
    case EngineChoice::Pruning: return "pruning";
    case EngineChoice::Mcts: return "mcts";
    case EngineChoice::MctsNn: return "mcts_nn";
    default: return "off";
    }
}

std::wstring fileNameOf(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    std::wstring result;
    result.reserve(name.size());
    for (unsigned char c : name) result.push_back(static_cast<wchar_t>(c));
    return result;
}

std::wstring engineChoiceLabel(EngineChoice choice) {
    switch (choice) {
    case EngineChoice::Pruning: return L"剪枝";
    case EngineChoice::Mcts: return L"MCTS";
    case EngineChoice::MctsNn: {
        std::wstring label = g_nnModelPath.empty() ? L"MCTS+NN" : L"MCTS+NN · " + fileNameOf(g_nnModelPath);
        if (g_engineAnalysis.backend == "cuda") label += L" (GPU)";
        else if (!g_engineAnalysis.backend.empty()) label += L" (CPU)";
        return label;
    }
    default: return L"未启用";
    }
}

std::wstring widen(const std::string& text) {
    std::wstring result;
    result.reserve(text.size());
    for (unsigned char c : text) result.push_back(static_cast<wchar_t>(c));
    return result;
}

bool sendEngine(const std::string& line) {
    if (!g_engineClient.running()) return false;
    return g_engineClient.send(line);
}

void clearEngineAnalysis() {
    g_engineAnalysis = EngineAnalysisView{};
}

// The engine sends one JSON object per `info` line.  An unparseable line is an
// error on screen, never a silently blank board.
void applyEngineAnalysis(const std::string& payload) {
    EngineAnalysisView view;
    std::string error;
    if (!parseAnalysis(payload, view, &error)) {
        g_engineMessage = L"引擎分析数据无法解析：" + widen(error);
        return;
    }
    g_engineAnalysis = std::move(view);
    g_engineMessage = g_enginePending ? L"引擎思考中…" : L"分析中…";
}

void startEngineAnalysis();

// Under the blocking rule a side can be left with no placement at all; the rules
// then force a pass.  Apply them here so every entry point (human click, built-in
// AI, engine move) ends up in the same state, and tell the engine about it.
void settleForcedPasses() {
    int guard = 0;
    bool passed = false;
    while (g_position.mustPass() && guard++ < 2) {
        const Player side = g_position.sideToMove();
        if (!g_position.pass()) break;
        passed = true;
        if (g_engineClient.running()) {
            sendEngine("stop");
            g_enginePending = false;
            sendEngine(std::string("play ") + (side == Player::Red ? "red" : "black") + " pass");
        }
    }
    if (passed) {
        g_engineMessage = L"无合法落点，强制停一手";
        if (g_position.gameOver()) {
            g_analysisOn = false;
            clearEngineAnalysis();
        }
    }
}

void notifyEnginePlay(Move move, Player mover) {
    if (!g_engineClient.running()) return;
    sendEngine("stop");
    g_enginePending = false;
    // A new position: the previous "budget spent, continue without limit" state no
    // longer applies.
    g_analysisFinished = false;
    if (isPass(move)) {
        sendEngine(std::string("play ") + (mover == Player::Red ? "red" : "black") + " pass");
    } else {
        sendEngine("play " + std::string(mover == Player::Red ? "red" : "black") + " " +
                   squareName(move.cell) + " " + std::to_string(static_cast<int>(move.number)));
    }
    if (!g_analysisOn) clearEngineAnalysis();
    if (g_analysisOn && !g_position.gameOver()) startEngineAnalysis();
}

// Start (or extend) the analysis of the current position.  `visits == 0` means
// no limit: the search keeps growing the same tree until Space stops it.  The
// previous display is deliberately kept until the first new info line arrives, so
// the marks do not blink on every refresh.
void startEngineAnalysis(int visits) {
    if (g_engineChoice == EngineChoice::Off || !g_engineClient.running() || g_position.gameOver()) return;
    g_enginePending = false;
    sendEngine("stop");
    g_analysisOn = true;
    g_analysisFinished = false;
    sendEngine("analyze " + std::string(g_position.sideToMove() == Player::Red ? "red" : "black") +
               " " + std::to_string(std::max(0, visits)));
    g_engineMessage = visits > 0 ? (L"分析中（上限 " + std::to_wstring(visits) + L" 访问，空格停止）")
                                 : L"分析中（不限访问，空格停止）";
    refresh();
}

void startEngineAnalysis() { startEngineAnalysis(g_analysisVisits); }

void stopEngineAnalysis() {
    g_analysisOn = false;
    if (g_engineClient.running()) sendEngine("stop");
    g_engineMessage = g_analysisFinished ? L"已达访问上限（空格继续，不再限量）"
                                         : L"分析已暂停（空格继续，树保留）";
    refresh();
}

// Space handler: start, stop, or continue past the limit.  Stopping keeps the
// numbers on screen -- they are still the best information about this position.
void toggleAnalysis() {
    if (g_position.gameOver()) return;
    if (g_analysisOn) {
        stopEngineAnalysis();
        return;
    }
    if (!g_engineReady || g_engineChoice == EngineChoice::Off) {
        g_engineMessage = L"先用 引擎 菜单选一个引擎";
        refresh();
        return;
    }
    // Continuing after a finished (budget-limited) run ignores the budget: the
    // visit limit is a stopping point, not a cap on what may be searched.
    startEngineAnalysis(g_analysisFinished ? 0 : g_analysisVisits);
}

void syncEnginePosition() {
    if (!g_engineClient.running()) return;
    g_enginePending = false;
    g_analysisFinished = false;
    sendEngine("stop");
    clearEngineAnalysis();
    sendEngine("clear_board");
    for (std::size_t i = 0; i < g_position.history().size(); ++i) {
        const Move move = g_position.history()[i];
        const Player mover = (i % 2 == 0) ? Player::Red : Player::Black;
        sendEngine("play " + std::string(mover == Player::Red ? "red" : "black") + " " +
                   squareName(move.cell) + " " + std::to_string(static_cast<int>(move.number)));
    }
    if (g_analysisOn && !g_position.gameOver()) startEngineAnalysis();
    g_engineMessage = L"引擎已同步";
}

void stopEngineProcess() {
    g_engineClient.stop();
    g_engineReady = false;
    g_analysisOn = false;
    g_enginePending = false;
    clearEngineAnalysis();
    g_engineMessage = L"外部引擎未启用";
}

// ANSI (what the *A APIs return) <-> UTF-8 (what argv/Utf8Args and the engine use).
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

// ---------------------------------------------------------------------------
// Model discovery
//
// The window can be started from anywhere (Explorer, a shortcut, build/), so
// "models/" relative to the current directory is not enough: that showed only
// the two hand-made fixtures and hid every trained network.  Instead scan a few
// roots -- the executable's own folder, its parent, the working directory, and
// anything in SHUYI_MODEL_DIRS -- and inside each root collect every *.bin that
// lives in a directory called "models".  Run names are prefixed so that two
// gen_000.bin from different runs stay distinguishable.
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

bool skipDirectory(const char* name);  // defined below; the root check needs it

std::vector<std::string> modelRoots() {
    std::vector<std::string> roots;
    const auto add = [&](const std::string& candidate) {
        if (candidate.empty()) return;
        const std::string full = normalizedPath(candidate);
        if (GetFileAttributesA(full.c_str()) == INVALID_FILE_ATTRIBUTES) return;
        // A build tree carries its own copy of the test fixtures, so scanning it
        // lists the same model twice under the same name.  Skip the root, not the
        // copy: the source tree one directory up is the one that ships.
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
// `relative` is the path below the scanned root, e.g. "runs/training/b4c32-nopass-20261124/models/x.bin".
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

// Authoritative: the ruleset is a byte in the model header, so read it instead
// of guessing from the file name (gen_000.bin says nothing).
// Training steps encoded in a checkpoint name: b4c32-s34760-d470705.bin -> 34760.
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
    // rule-set byte is in the same place.  Without this case a v4 model reads as
    // "unknown" and the menu silently drops it (it only lists models that match
    // the active rule set), which is exactly how the package ended up with no
    // model at all after the A-conditioned network shipped.
    const bool v4 = std::memcmp(header, "SHUYINN4", 8) == 0;
    const bool v3 = std::memcmp(header, "SHUYINN3", 8) == 0;
    const bool v2 = std::memcmp(header, "SHUYINN2", 8) == 0;
    if (!v3 && !v4 && !v2) return -1;
    if (v2) return 0;  // v2 has no ruleset byte; those weights are classic
    if (got < 33) return 0;
    // magic(8) version(4) channels(4) blocks(4) inputs(4) globals(4) flags(4) ruleset(1)
    const int ruleset = header[32];
    // Not `== 1 ? 1 : 0`: that maps a nopass model (2) onto classic, and the menu
    // then hides it, because it only lists models compatible with the rule set in
    // force.  An unknown byte is "not a model this build understands".
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
    // Build trees and the package directory contain copies of the same models;
    // caches are noise; runs/archive holds finished experiments whose weights are
    // history, not candidates.
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
    // FNV-1a over the bytes.  Two copies of one weights file (the package next to
    // the exe, the source tree, a server mirror) must not become two menu entries
    // that look like a choice between different networks.
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
    // The same model is reachable through several roots (the exe directory, its
    // parent, the current directory) and is often copied byte for byte into more
    // than one of them.  Keep one entry per file path and one per distinct file
    // content.
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
            // Keep the shallower copy: models/x.bin is the name the package ships
            // and the one a user recognises, runs/<run>/models/x.bin is the same
            // bytes under the run that produced them.
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

// Display label for a rule set.  Kept in the front end, not in core: core is
// included from CUDA translation units, and a Chinese string literal there
// breaks nvcc's own parser on Windows.
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

// Fallback when nothing has been picked from the menu yet: the model with the
// most training steps that was trained for the ruleset currently in force.  A
// model from an incompatible rule set is never selected -- the engine would
// refuse it, and silently running the wrong network is worse than having none.
std::string defaultModelPath() {
    char buffer[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableA("SHUYI_NN_MODEL", buffer, MAX_PATH);
    if (length > 0 && length < MAX_PATH) return ansiToUtf8(std::string(buffer, length));
    if (g_modelList.empty()) scanModels();
    const ModelEntry* best = nullptr;
    for (const ModelEntry& entry : g_modelList) {
        // `compatible`, not equal: a blocking-trained model is a valid choice
        // under NoPass (every NoPass game is a blocking game up to its end).
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
        const bool picked = g_engineChoice == EngineChoice::MctsNn && static_cast<int>(id) - ID_MODEL_BASE == index;
        CheckMenuItem(g_modelMenu, id, MF_BYCOMMAND | (picked ? MF_CHECKED : MF_UNCHECKED));
    }
}

// Selecting a model both records the path and forces the engine process to
// restart with it (set_model is only read at startup).
void setEngineChoice(EngineChoice choice);

void selectModel(int index) {
    if (index < 0 || index >= static_cast<int>(g_modelList.size())) return;
    const ModelEntry& entry = g_modelList[index];
    // A network trained for an incompatible rule set is not a valid choice: the
    // engine would refuse it later, so say it here while the reason is obvious.
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
// Engine program setting
//
// The front end and the engine are separate programs that may come from
// different projects; the only contract is the line protocol.  So the engine is
// a *path the user chooses*, stored next to the window, and remembered:
//   cmake --install ... --component engine   -> some folder
//   引擎 -> 选择引擎程序...                  -> point the UI at it
// Everything else (discovery, preferring a CUDA build) is a convenience fallback
// for the case where nothing has been chosen yet.
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
    while (begin < end && (text[begin] == ' ' || text[begin] == '	')) ++begin;
    while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '	')) --end;
    return text.substr(begin, end - begin);
}

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

// Plain `key=value` in UTF-8.  The Win32 profile API would read the file as ANSI
// and mangle any non-ASCII path, which is exactly what happens to this project's
// folder name, so the file is parsed here instead.
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
                g_aValue = std::clamp(std::atof(line.substr(8).c_str()), -40.0, 40.0);
            } else if (line.rfind("ruleset=", 0) == 0) {
                Ruleset parsed = g_ruleset;
                if (parseRuleset(trimAscii(line.substr(8)), parsed)) g_ruleset = parsed;
            }
        }
    }
    // A shipped package writes `engine=shuyi_engine.exe`: a relative path is
    // resolved against this executable, not against wherever it was launched from.
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

// ---------------------------------------------------------------------------
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
        // Several compatible engines: the GPU one is the one to run.
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
        g_engineMessage = L"没有找到可用的引擎：引擎菜单 → 选择引擎程序…";
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

    // A rejected command has to stop the start-up: otherwise the window would
    // sit on an empty analysis and the real cause would never be visible.
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
            // A package without weights is a supported download, so say what to do
            // instead of starting an engine that cannot analyse anything.
            failure = L"没有可用模型：把 b<blocks>c<channels>-s<steps>-d<data>.bin 放进 "
                      L"models/（或引擎菜单 → MCTS + 网络 → 重新扫描）";
        } else {
            // The model must be accepted before the engine that needs it is selected.
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
        // `set_model` + `set_engine` both succeeded, but "ready" is the engine's
        // own answer to "can you search with this network": ask it, once.
        std::string capabilities;
        if (!g_engineClient.command("capabilities", &capabilities, 3000) ||
            capabilities.find("model=1") == std::string::npos) {
            failure = L"引擎没有加载到模型：\n" + widen(g_nnModelPath) +
                      L"\n\n请在 引擎 → MCTS + 网络 里选择与当前规则一致的模型。";
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
    const std::wstring engineFile = g_engineClient.path();
    const std::size_t slash = engineFile.find_last_of(L"\\/");
    const std::wstring shortName =
        slash == std::wstring::npos ? engineFile : engineFile.substr(slash + 1);
    g_engineMessage = L"引擎已启动：" + shortName +
                      (probe.backend == "cuda" ? L" (CUDA)" : L" (CPU)");
    syncEnginePosition();
    return true;
}

// ---------------------------------------------------------------- settings ---
//
// A visit budget, not a wall clock: "512 visits" means the same search on the
// laptop and on the GPU.  Both numbers are persisted in shuyi-ui.ini and sent to
// the engine when it starts (and whenever they change).
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
        // A is a real number now: 1.5 is the shipped value because a half-integer
        // cannot draw (raw + 1.5 is never 0), while the rounded 1 turns every
        // P = -1 position into a draw.
        const auto readA = [&](HWND edit, double fallback) {
            if (!edit) return fallback;
            GetWindowTextW(edit, buffer, 32);
            wchar_t* end = nullptr;
            const double value = std::wcstod(buffer, &end);
            if (end == buffer || !std::isfinite(value)) return fallback;
            return std::clamp(value, -40.0, 40.0);
        };
        const int analysis = readInt(g_settingsAnalysisEdit, g_analysisVisits, 0, 50000000);
        const int play = readInt(g_settingsPlayEdit, g_playVisits, 1, 50000000);
        const double aValue = readA(g_settingsAValueEdit, g_aValue);
        g_analysisVisits = analysis;
        g_playVisits = play;
        saveSettings();
        if (g_engineClient.running()) {
            sendEngine("set_visits " + std::to_string(g_analysisVisits));
        }
        if (aValue != g_aValue) setAValue(aValue);
        g_engineMessage = L"设置已保存（分析 " + std::to_wstring(g_analysisVisits) + L" / 落子 " +
                          std::to_wstring(g_playVisits) + L" 访问）";
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
    const int height = scale(228);
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
    label(L"A值（目数，可为负）", top);
    g_settingsAValueEdit = editText(top, formatAValueW(g_aValue));
    top += rowHeight + scale(6);
    label(L"空格：开始 / 停止 / 超限后继续搜索", top);

    const int buttonTop = height - scale(58);
    HWND ok = CreateWindowExW(0, L"BUTTON", L"确定",
                              WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON, scale(196),
                              buttonTop, scale(72), scale(28), g_settingsWnd,
                              reinterpret_cast<HMENU>(static_cast<INT_PTR>(ID_SETTINGS_OK)),
                              g_instance, nullptr);
    HWND cancel = CreateWindowExW(0, L"BUTTON", L"取消", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                                  scale(274), buttonTop, scale(72), scale(28), g_settingsWnd,
                                  reinterpret_cast<HMENU>(static_cast<INT_PTR>(ID_SETTINGS_CANCEL)),
                                  g_instance, nullptr);
    for (HWND child : {ok, cancel}) SendMessageW(child, WM_SETFONT, reinterpret_cast<WPARAM>(g_fontUi), TRUE);
    for (HWND child : {g_settingsAnalysisEdit, g_settingsPlayEdit, g_settingsAValueEdit})
        SendMessageW(child, EM_SETSEL, 0, -1);

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
    g_engineMessage = L"引擎已指定：" + std::wstring(chosen);
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
    g_engineMessage = L"引擎路径已清空，改为自动探测";
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
    text += L"\n每步时间：650 ms";
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
    if (aiToMove()) maybeAiMove();
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
        if (g_analysisOn || g_enginePending) applyEngineAnalysis(line.substr(5));
        return;
    }
    if (line.rfind("= done", 0) == 0) {
        // The analysis ran out of budget on its own.  The numbers stay on screen
        // and the tree stays in the engine, so Space can extend it without a
        // limit.
        g_analysisOn = false;
        g_analysisFinished = true;
        g_engineMessage = L"已达访问上限（空格继续，不再限量）";
        refresh();
        return;
    }
    if (line.rfind("= stopped", 0) == 0) {
        // We asked for this; the last snapshot is still the best information.
        return;
    }
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
        // The engine found no placement: let the rules settle the forced pass
        // and keep the engine in step.
        g_enginePending = false;
        g_pendingReplyId.clear();
        g_engineMessage = L"引擎过手（无合法落点）";
        settleForcedPasses();
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
    const Player mover = g_position.sideToMove();
    g_position.play(move);
    sendEngine("play " + std::string(mover == Player::Red ? "red" : "black") + " " +
               squareName(move.cell) + " " + std::to_string(number));
    g_thinking = false;
    if (humanToMove()) g_selected = firstAvailable(g_position.sideToMove());
    g_engineMessage = L"引擎已落子";
    settleForcedPasses();
    if (g_position.gameOver()) {
        g_analysisOn = false;
        clearEngineAnalysis();
    }
    if (g_analysisOn && !g_position.gameOver()) startEngineAnalysis();
    refresh();
    if (aiToMove()) maybeAiMove();
}

void drainEngineLines() {
    for (const std::string& line : g_engineClient.drain()) handleEngineLine(line);
}

// (the Space handler `toggleAnalysis` is defined next to startEngineAnalysis)

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
    g_position.setRuleset(ruleset);
    if (g_engineChoice != EngineChoice::Off) {
        // Restart the engine process so it picks up the new rule set.
        g_analysisOn = false;
        stopEngineProcess();
        if (!startEngineProcess()) {
            MessageBoxW(g_hwnd, g_engineMessage.c_str(), L"引擎", MB_OK | MB_ICONWARNING);
        }
    }
    clearEngineAnalysis();
    syncRulesetChecks();
    refresh(true);
    return true;
}

void syncEngineChecks() {
    if (!g_engineMenu) return;
    int id = ID_ENGINE_OFF;
    if (g_engineChoice == EngineChoice::Pruning) id = ID_ENGINE_PRUNING;
    else if (g_engineChoice == EngineChoice::Mcts) id = ID_ENGINE_MCTS;
    CheckMenuRadioItem(g_engineMenu, ID_ENGINE_OFF, ID_ENGINE_MCTS, id, MF_BYCOMMAND);
    // The model submenu is its own group: it only means something for MCTS+NN.
    if (g_modelMenuPosition >= 0) {
        CheckMenuItem(g_engineMenu, static_cast<UINT>(g_modelMenuPosition),
                      MF_BYPOSITION | (g_engineChoice == EngineChoice::MctsNn ? MF_CHECKED : MF_UNCHECKED));
    }
    syncModelChecks();
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
struct Layout {
    RECT client{};
    RECT board{};
    RECT panel{};
    int cell = 60;
};

struct PanelRects {
    RECT turn{};
    RECT stats{};
    RECT chips[2][9]{};
    RECT mode{};
    RECT analysis{};
    RECT newButton{};
    RECT undoButton{};
    RECT aMinus{};
    RECT aPlus{};
};

Layout computeLayout(HWND hwnd) {
    Layout result;
    GetClientRect(hwnd, &result.client);
    const int width = result.client.right;
    const int height = result.client.bottom;
    const int margin = scale(22);
    const int coord = scale(30);
    const int statusHeight = scale(34);
    const int panelWidth = scale(318);
    const int gap = scale(24);

    int availableW = width - 2 * margin - 2 * coord - gap - panelWidth;
    int availableH = height - 2 * margin - statusHeight - 2 * coord;
    int cell = std::min(availableW / 9, availableH / 9);
    cell = std::max(scale(36), std::min(cell, scale(74)));
    result.cell = cell;
    const int boardSize = cell * 9;
    result.board = {margin + coord, margin + coord, margin + coord + boardSize,
                    margin + coord + boardSize};
    result.panel = {result.board.right + gap, margin, width - margin, height - margin - statusHeight};
    return result;
}

PanelRects computePanelRects(const Layout& layout) {
    PanelRects r;
    const int x = layout.panel.left;
    const int y = layout.panel.top;
    const int w = layout.panel.right - layout.panel.left;
    const int pad = scale(16);
    int cursor = y + scale(92);

    r.turn = {x + pad, cursor, x + w - pad, cursor + scale(54)};
    cursor += scale(68);
    r.stats = {x + pad, cursor, x + w - pad, cursor + scale(94)};
    cursor += scale(132);

    const int labelX = x + pad + scale(42);
    const int chipWidth = (x + w - pad - labelX) / 9;
    const int chipHeight = scale(46);
    const int rowGap = scale(8);
    for (int side = 0; side < 2; ++side) {
        for (int n = 0; n < 9; ++n) {
            r.chips[side][n] = {labelX + n * chipWidth, cursor + side * (chipHeight + rowGap),
                                labelX + (n + 1) * chipWidth - scale(2),
                                cursor + side * (chipHeight + rowGap) + chipHeight};
        }
    }
    cursor += 2 * chipHeight + rowGap + scale(22);
    r.mode = {x + pad, cursor, x + w - pad, cursor + scale(24)};
    const int buttonY = layout.panel.bottom - scale(54);
    r.analysis = {x + pad, r.mode.bottom + scale(10), x + w - pad, buttonY - scale(12)};
    const int buttonGap = scale(10);
    const int buttonW = (w - 2 * pad - buttonGap) / 2;
    r.newButton = {x + pad, buttonY, x + pad + buttonW, buttonY + scale(40)};
    r.undoButton = {r.newButton.right + buttonGap, buttonY, x + w - pad, buttonY + scale(40)};

    r.aMinus = {r.stats.right - scale(58), r.stats.top + scale(38),
                r.stats.right - scale(34), r.stats.top + scale(62)};
    r.aPlus = {r.stats.right - scale(28), r.stats.top + scale(38),
               r.stats.right - scale(4), r.stats.top + scale(62)};
    return r;
}

RECT overlayRect(const Layout& layout) {
    const int boardSize = layout.board.right - layout.board.left;
    const int w = std::min(scale(430), boardSize - scale(36));
    const int h = scale(184);
    const int cx = layout.board.left + boardSize / 2;
    const int cy = layout.board.top + boardSize / 2;
    return {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
}

RECT overlayButton(const Layout& layout) {
    RECT box = overlayRect(layout);
    const int w = scale(132);
    const int h = scale(38);
    const int top = box.bottom - scale(50);
    return {(box.left + box.right - w) / 2, top, (box.left + box.right + w) / 2, top + h};
}

bool contains(const RECT& rect, POINT point) {
    return point.x >= rect.left && point.x < rect.right && point.y >= rect.top && point.y < rect.bottom;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
// Pieces are square: the board is a grid of claims, not a shape-sorting game, and
// a square footprint lets the analysis marks reuse exactly the same geometry.
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


// ------------------------------------------------------------------ marks ---
//
// One mark per analysed move, drawn with the same square footprint as a piece:
// a thin outline, a translucent-looking fill (mixed against the board colour, so
// there is no alpha channel to worry about) and three rows of text.
//
//   row 1  win rate
//   row 2  visits, abbreviated to at most four characters
//   row 3  expected terminal score
//
// Colour is relative to the best move, not absolute: the best is light blue,
// everything else runs red (far behind) -> green (close).  Marks more than 20
// points of win rate behind, or with under 5% of the best move's visits, are not
// drawn at all -- they are noise on a 9x9 board.
constexpr double kMarkWinRateWindow = 0.20;
constexpr double kMarkVisitFloor = 0.05;
// Also draw at least this many of the best marks even when one move collected
// nearly all visits.  Without it a converged search leaves a single mark on the
// board, which reads as "the engine proved a unique answer" -- it did not.
constexpr int kMarkAlwaysShow = 8;

// Remembers the marked cell so ties do not alternate on every refresh.
int g_markedBestCell = -1;

// The engine's recommended move, as the cell the board can mark.  The mark and
// the summary must describe this one cell: the engine picks the move, and a front
// end that ranks cells on its own can name a different one (two symmetric squares
// are one move, and the panel used to say "best d6-1" while d4-1 was highlighted).
const CellAnalysisView* analysisBestCell() {
    const Move best = g_engineAnalysis.best;
    if (!best.valid() || isPass(best)) return nullptr;
    if (best.cell < 0 || best.cell >= kCells) return nullptr;
    if (!g_engineAnalysis.hasCell[best.cell]) return nullptr;
    const CellAnalysisView& info = g_engineAnalysis.cells[best.cell];
    return info.hasData ? &info : nullptr;
}

COLORREF markColor(double relative, bool best) {
    if (best) return RGB(126, 196, 255); // light blue
    const double t = std::clamp(1.0 + relative / kMarkWinRateWindow, 0.0, 1.0);
    const COLORREF red = RGB(214, 92, 82);
    const COLORREF green = RGB(96, 186, 118);
    return mixColor(red, green, t);
}

void drawAnalysisMarks(HDC dc, const Layout& layout) {
    const int cell = layout.cell;
    const RECT& board = layout.board;

    // A solved cell is an exact result, so it is ranked by its win rate; an
    // estimate is ranked by its Wilson lower bound.  A solved 100% in one visit
    // is therefore never outranked by a half-explored 55%, and a solved line is
    // never hidden by the visit floor below.
    const auto cellScore = [](const CellAnalysisView& info) {
        return info.proven ? info.winrate : winRateLowerBound(info.winrate, info.visits);
    };
    long long bestVisits = 0;
    double markedScore = -1.0;
    for (int cellIndex = 0; cellIndex < kCells; ++cellIndex) {
        if (!g_engineAnalysis.hasCell[cellIndex]) continue;
        const auto& info = g_engineAnalysis.cells[cellIndex];
        if (!info.hasData) continue;
        bestVisits = std::max<long long>(bestVisits, info.visits);
        markedScore = std::max(markedScore, cellScore(info));
    }
    if (markedScore < 0.0) return;

    // The panel names the engine's recommended move, so the mark goes there too.
    // It used to be chosen independently by the confidence bound, which lands on
    // whichever cell of an equivalence class happens to rank first: the board then
    // highlighted d4-1 while the panel said "best d6-1" (the same move under the
    // board's symmetry), and the two numbers could not agree because the panel
    // measured the bound over the root's visits and the cell over its own.
    int bestCell = -1;
    if (const CellAnalysisView* chosen = analysisBestCell()) {
        bestCell = chosen->move.cell;
    }
    double bestScore = -1.0;
    if (bestCell >= 0) {
        bestScore = cellScore(g_engineAnalysis.cells[bestCell]);
    }
    for (int cellIndex = 0; cellIndex < kCells && bestCell < 0; ++cellIndex) {
        if (g_position.stoneAt(cellIndex) != 0) continue;  // only empty cells can be played
        if (!g_engineAnalysis.hasCell[cellIndex]) continue;
        const auto& info = g_engineAnalysis.cells[cellIndex];
        // A solved cell is a result even with zero visits: the exact endgame
        // solver answers without sampling, so requiring visits here dropped every
        // cell of a solved position.
        if (!info.hasData || (info.visits <= 0 && !info.proven)) continue;
        if (cellScore(info) > bestScore) {
            bestCell = cellIndex;
            bestScore = cellScore(info);
        }
    }
    g_markedBestCell = bestCell;
    if (bestCell < 0) return;

    const int radius = static_cast<int>(cell * 0.40);
    const int corner = std::max(2, radius / 4);
    // Three text rows share the mark's height.
    const int rowHeight = std::max(8, (radius * 2) / 3);
    const int textTop = -radius;
    // Rank first: the top few marks are drawn no matter what, because a search
    // that pours almost every visit into one move would otherwise leave a board
    // with a single mark and look like "the engine decided".
    std::array<int, kCells> rank{};
    rank.fill(-1);
    {
        std::vector<int> order;
        for (int cellIndex = 0; cellIndex < kCells; ++cellIndex) {
            if (!g_engineAnalysis.hasCell[cellIndex]) continue;
            if (!g_engineAnalysis.cells[cellIndex].hasData) continue;
            order.push_back(cellIndex);
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return cellScore(g_engineAnalysis.cells[a]) > cellScore(g_engineAnalysis.cells[b]);
        });
        for (std::size_t i = 0; i < order.size(); ++i) rank[order[i]] = static_cast<int>(i);
    }
    for (int cellIndex = 0; cellIndex < kCells; ++cellIndex) {
        if (g_position.stoneAt(cellIndex) != 0) continue;
        if (!g_engineAnalysis.hasCell[cellIndex]) continue;
        const auto& info = g_engineAnalysis.cells[cellIndex];
        if (!info.hasData) continue;
        const bool topMark = rank[cellIndex] >= 0 && rank[cellIndex] < kMarkAlwaysShow;
        const double score = cellScore(info);
        const double relative = score - markedScore;
        // A solved cell is a fact, not a sample: draw it even when its win rate is
        // far below the best, because the number it carries is the margin.  The
        // window exists to keep estimates from crowding the board.
        if (!topMark && !info.proven && relative < -kMarkWinRateWindow) continue;
        if (!topMark && !info.proven && bestVisits > 0 &&
            info.visits < static_cast<long long>(bestVisits * kMarkVisitFloor)) {
            continue;
        }
        const bool best = cellIndex == bestCell;
        const COLORREF colour = markColor(relative, best);
        const int cx = board.left + xOf(cellIndex) * cell + cell / 2;
        const int cy = board.top + (8 - yOf(cellIndex)) * cell + cell / 2;
        const RECT mark{cx - radius, cy - radius, cx + radius, cy + radius};
        roundRect(dc, mark, corner, mixColor(kBoard, colour, 0.62), colour, std::max(1, scale(1)));
        int y = cy + textTop;
        // A solved cell shows its exact margin instead of a win rate: "loses by 2"
        // and "loses by 18" are both 0% and would read identically.  An estimate
        // shows the Wilson lower bound, so a single lucky sample is never printed
        // as "100%" -- that reads like a proof, and only `proven` is a proof.
        // Row 1 is always the win rate: a solved cell is exactly 0 / 50 / 100, an
        // estimate is its Wilson lower bound (so one lucky sample is never printed
        // as 100% -- only `proven` is a proof).
        drawTextAt(dc, formatWinRate(cellScore(info)), cx - radius, y, radius * 2, rowHeight,
                   g_fontSmall, RGB(28, 32, 38));
        y += rowHeight;
        // Row 2 is the visit count, and a solved cell has none: the answer did not
        // come from sampling, so "0" would read as an empty search.  The row stays
        // reserved so the margin sits on the same line on every cell.
        if (info.visits > 0) {
            drawTextAt(dc, formatVisits(info.visits), cx - radius, y, radius * 2, rowHeight,
                       g_fontSmall, RGB(28, 32, 38));
        }
        y += rowHeight;
        // Row 3 is the margin: exact (whole points) when solved, an average from a
        // score head otherwise, and "--" when no visit carried a real score.
        std::wstring scoreText = L"--";
        if (info.hasScore) {
            scoreText = info.proven ? formatPoints(info.averageScore)
                                    : formatAverageScore(info.averageScore);
        }
        drawTextAt(dc, scoreText, cx - radius, y, radius * 2, rowHeight, g_fontSmall,
                   RGB(28, 32, 38));
    }
}

void drawBoard(HDC dc, const Layout& layout) {
    const int cell = layout.cell;
    const RECT& board = layout.board;

    fillRectColor(dc, board, kBoard);
    // Subtle vignette-like border.
    for (int i = 0; i < scale(5); ++i) {
        RECT edge{board.left + i, board.top + i, board.right - i, board.bottom - i};
        HBRUSH brush = CreateSolidBrush(i == 0 ? kBoardLine : kBoard);
        FrameRect(dc, &edge, brush);
        DeleteObject(brush);
    }

    // The analysis overlay is drawn after the grid so the marks sit on top of the
    // lines, exactly like a piece would.
    const bool showHeat = g_engineAnalysis.valid && g_engineChoice != EngineChoice::Off;

    for (int i = 0; i <= 9; ++i) {
        const int x = board.left + i * cell;
        const int y = board.top + i * cell;
        line(dc, x, board.top, x, board.bottom, kBoardLine, i == 0 || i == 9 ? 2 : 1);
        line(dc, board.left, y, board.right, y, kBoardLine, i == 0 || i == 9 ? 2 : 1);
    }

    if (showHeat) drawAnalysisMarks(dc, layout);

    const int stars[5][2] = {{2, 2}, {6, 2}, {4, 4}, {2, 6}, {6, 6}};
    for (const auto& star : stars) {
        const int cx = board.left + star[0] * cell + cell / 2;
        const int cy = board.top + (8 - star[1]) * cell + cell / 2;
        filledCircle(dc, cx, cy, std::max(2, scale(3)), kBoardLine);
    }

    // Coordinates.
    for (int i = 0; i < 9; ++i) {
        const std::wstring letter(1, static_cast<wchar_t>(L'a' + i));
        const std::wstring number = std::to_wstring(9 - i);
        const int cx = board.left + i * cell + cell / 2;
        drawTextAt(dc, letter, cx - cell / 2, board.top - scale(28), cell, scale(22), g_fontCoord,
                   kMuted);
        drawTextAt(dc, letter, cx - cell / 2, board.bottom + scale(6), cell, scale(22), g_fontCoord,
                   kMuted);
        const int cy = board.top + i * cell + cell / 2;
        drawTextAt(dc, number, board.left - scale(30), cy - cell / 2, scale(22), cell, g_fontCoord,
                   kMuted);
        drawTextAt(dc, number, board.right + scale(8), cy - cell / 2, scale(22), cell, g_fontCoord,
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

    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 9; ++x) {
            const int cellIndex = index(x, y);
            const int stone = g_position.stoneAt(cellIndex);
            if (stone == 0) continue;
            const int cx = board.left + x * cell + cell / 2;
            const int cy = board.top + (8 - y) * cell + cell / 2;
            const int radius = static_cast<int>(cell * 0.37);
            drawStone(dc, cx, cy, radius, stone > 0, stone > 0 ? stone : -stone);
            if (!g_position.history().empty() && g_position.lastMove().cell == cellIndex) {
                const int grow = scale(3);
                const RECT ring{cx - radius - grow, cy - radius - grow, cx + radius + grow,
                                cy + radius + grow};
                HPEN pen = CreatePen(PS_SOLID, std::max(2, scale(2)), kGold);
                HGDIOBJ old = SelectObject(dc, pen);
                HGDIOBJ oldBrush = SelectObject(dc, GetStockObject(NULL_BRUSH));
                RoundRect(dc, ring.left, ring.top, ring.right, ring.bottom, std::max(3, radius / 3),
                          std::max(3, radius / 3));
                SelectObject(dc, oldBrush);
                SelectObject(dc, old);
                DeleteObject(pen);
            }
        }
    }

    // The best move is highlighted by its mark (light blue), so no extra ring: two
    // overlapping highlights on a 9x9 board read as noise.
}

void drawEngineAnalysis(HDC dc, const PanelRects& rects) {
    const RECT& box = rects.analysis;
    if (box.bottom - box.top < scale(54)) return;
    roundRect(dc, box, scale(9), kPanelDeep, RGB(56, 64, 78));
    const int left = box.left + scale(12);
    const int width = box.right - box.left - scale(24);
    int y = box.top + scale(7);

    std::wstring title = L"引擎分析 · ";
    title += engineChoiceLabel(g_engineChoice);
    if (!g_engineAnalysis.nn.empty() && g_engineAnalysis.nn != "none") {
        title += L" · ";
        title += widen(g_engineAnalysis.nn);
    }
    drawTextAt(dc, title, left, y, width, scale(22), g_fontSmall, kCyan,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    y += scale(23);

    if (g_engineChoice == EngineChoice::Off) {
        drawTextAt(dc, L"在“引擎”菜单选择搜索器", left, y, width, scale(20), g_fontSmall, kMuted,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        return;
    }
    if (g_enginePending) {
        drawTextAt(dc, L"搜索中… 10Hz 刷新", left, y, width, scale(20), g_fontSmall, kGold,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        return;
    }
    if (!g_engineAnalysis.valid) {
        drawTextAt(dc, g_analysisOn ? L"等待分析数据…" : L"按空格开始分析",
                   left, y, width, scale(20), g_fontSmall, kMuted,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        return;
    }

    const auto& analysis = g_engineAnalysis;
    const std::wstring best = L"最佳 " + widen(squareName(analysis.best.cell)) + L"-" +
                              std::to_wstring(static_cast<int>(analysis.best.number));
    drawTextAt(dc, best, left, y, width, scale(22), g_fontUi, kCyan,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    y += scale(23);
    std::wstring metric;
    if (analysis.kind.rfind("mcts", 0) == 0) {
        // Describe the same cell the board marks, with that cell's own sample
        // count.  Measuring the bound over the root's visits (which is what this
        // panel used to do) prints a number the board can never agree with: the
        // recommended move has its own visit count, and a bound over 512 visits
        // is not the bound over 96.
        const CellAnalysisView* bestCell = analysisBestCell();
        const bool proven = bestCell ? bestCell->proven : analysis.proven;
        const double winrate = bestCell ? bestCell->winrate : analysis.winrate;
        const long long visits = bestCell ? bestCell->visits : analysis.visits;
        const double shown = proven ? winrate : winRateLowerBound(winrate, visits);
        const bool hasScore = bestCell ? bestCell->hasScore : analysis.hasScore;
        const double score = bestCell ? bestCell->averageScore : analysis.averageScore;
        // An exact solve reports no visits: printing "0/0" would read as a search
        // that did nothing, when in fact it needs no search at all.
        const bool solvedExactly = analysis.visits == 0 && proven;
        metric = solvedExactly ? L"精确解算"
                               : L"访问 " + std::to_wstring(visits) + L"/" +
                                     std::to_wstring(analysis.visits);
        metric += L" · 胜率 " +
                  std::to_wstring(static_cast<int>(std::lround(shown * 100.0))) + L"%";
        metric += hasScore ? (proven ? L" · 目差 " : L" · 均分 ") : L" · 均分 --";
        if (hasScore) metric += signedDecimal(score);
        // A solved move stops searching on purpose: say so instead of looking
        // like a search that ran out of budget.
        if (proven && !solvedExactly) metric += L" · 已解算";
    } else {
        metric = L"深度 " + std::to_wstring(analysis.depth) + L" · 节点 " +
                 std::to_wstring(analysis.nodes) + L" · 评分 " +
                 signedText(analysis.score);
        if (analysis.window > 0 || analysis.pvsResearches > 0) {
            metric += L" · 窗±" + std::to_wstring(analysis.window) + L"/" +
                      std::to_wstring(analysis.windowResearches) + L" PVS " +
                      std::to_wstring(analysis.pvsResearches);
        }
    }
    drawTextAt(dc, metric, left, y, width, scale(20), g_fontSmall, kMuted,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    y += scale(21);
    // Say why the search is idle: the numbers stay on screen after Space stops it,
    // so the panel has to explain what the next Space will do.
    if (!g_analysisOn) {
        const wchar_t* hint = g_analysisFinished ? L"已达访问上限 · 空格继续（不限量，树保留）"
                                                 : L"已暂停 · 空格继续（树保留）";
        drawTextAt(dc, hint, left, y, width, scale(20), g_fontSmall,
                   g_analysisFinished ? kGold : kMuted,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        y += scale(20);
    }

    if (analysis.kind.rfind("mcts", 0) == 0 && !analysis.candidates.empty()) {
        const int rows = std::min<int>(4, static_cast<int>(analysis.candidates.size()));
        for (int i = 0; i < rows; ++i) {
            const auto& candidate = analysis.candidates[i];
            const double rate = std::clamp(candidate.winrate, 0.0, 1.0);
            const int barTop = y + i * scale(20);
            RECT bar{left + scale(58), barTop + scale(6), box.right - scale(16), barTop + scale(17)};
            roundRect(dc, bar, scale(3), RGB(55, 62, 75));
            RECT fill = bar;
            fill.right = fill.left + static_cast<int>((bar.right - bar.left) * rate);
            roundRect(dc, fill, scale(3), i == 0 ? kCyan : RGB(76, 140, 176));
            const std::wstring label = widen(squareName(candidate.move.cell)) + L"-" +
                                       std::to_wstring(static_cast<int>(candidate.move.number));
            drawTextAt(dc, label, left, barTop, scale(54), scale(20), g_fontSmall,
                       i == 0 ? kText : kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        }
    } else {
        std::wstring pv = L"PV " + widen(analysis.pv);
        if (pv.size() > 42) pv = pv.substr(0, 41) + L"…";
        drawTextAt(dc, pv, left, y, width, scale(20), g_fontSmall, kMuted,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    }
}

void drawPanel(HDC dc, const Layout& layout) {
    PanelRects rects = computePanelRects(layout);
    const int x = layout.panel.left;
    const int y = layout.panel.top;
    const int w = layout.panel.right - layout.panel.left;
    const int h = layout.panel.bottom - layout.panel.top;
    roundRect(dc, {x, y, x + w, y + h}, scale(14), kPanel, RGB(58, 64, 76), 1);

    drawTextAt(dc, L"数弈", x + scale(18), y + scale(14), w - scale(36), scale(46), g_fontTitle,
               kText, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, L"SHUYI · 数字落子棋", x + scale(20), y + scale(56), w - scale(40), scale(24),
               g_fontSubtitle, kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    line(dc, x + scale(16), y + scale(86), x + w - scale(16), y + scale(86), RGB(58, 64, 76));

    const Player side = g_position.sideToMove();
    roundRect(dc, rects.turn, scale(9), kPanelSoft);
    COLORREF sideColor = side == Player::Red ? kRed : kBlack;
    if (side == Player::Black) sideColor = RGB(15, 17, 22);
    filledCircle(dc, rects.turn.left + scale(24), (rects.turn.top + rects.turn.bottom) / 2,
                 scale(9), sideColor, side == Player::Black ? kBlackSoft : RGB(244, 164, 150), 1);
    drawTextAt(dc, turnLabel(), rects.turn.left + scale(42), rects.turn.top,
               rects.turn.right - rects.turn.left - scale(50), rects.turn.bottom - rects.turn.top,
               g_fontTurn, kText, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    roundRect(dc, rects.stats, scale(9), kPanelDeep);
    const int raw = g_position.rawScore();
    const double total = raw + g_aValue;
    const int sx = rects.stats.left + scale(12);
    const int sw = rects.stats.right - rects.stats.left - scale(24);
    drawTextAt(dc, L"当前格子分", sx, rects.stats.top + scale(5), sw - scale(70), scale(24),
               g_fontSmall, kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, signedText(raw), sx + sw - scale(76), rects.stats.top + scale(5), scale(76),
               scale(24), g_fontBig, raw > 0 ? kRedSoft : (raw < 0 ? RGB(164, 172, 188) : kText),
               DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, L"A值", sx, rects.stats.top + scale(35), sw - scale(100), scale(24), g_fontSmall,
               kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, signedValue(g_aValue), sx + sw - scale(96), rects.stats.top + scale(35), scale(36),
               scale(24), g_fontUi, kGold, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    roundRect(dc, rects.aMinus, scale(5), kPanelSoft, RGB(68, 75, 88));
    roundRect(dc, rects.aPlus, scale(5), kPanelSoft, RGB(68, 75, 88));
    drawTextAt(dc, L"−", rects.aMinus.left, rects.aMinus.top, rects.aMinus.right - rects.aMinus.left,
               rects.aMinus.bottom - rects.aMinus.top, g_fontUi, kText);
    drawTextAt(dc, L"+", rects.aPlus.left, rects.aPlus.top, rects.aPlus.right - rects.aPlus.left,
               rects.aPlus.bottom - rects.aPlus.top, g_fontUi, kText);
    drawTextAt(dc, L"若现在终局", sx, rects.stats.top + scale(65), sw - scale(80), scale(24),
               g_fontSmall, kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, signedValue(total), sx + sw - scale(76), rects.stats.top + scale(63), scale(76),
               scale(26), g_fontUi, total > 0 ? kRedSoft : (total < 0 ? RGB(174, 181, 195) : kGreen),
               DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

    drawTextAt(dc, L"剩余棋子", x + scale(16), rects.chips[0][0].top - scale(28), w - scale(32),
               scale(22), g_fontSmall, kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    for (int sideIndex = 0; sideIndex < 2; ++sideIndex) {
        const Player player = sideIndex == 0 ? Player::Red : Player::Black;
        const bool rowActive = g_position.sideToMove() == player && !g_position.gameOver();
        drawTextAt(dc, sideIndex == 0 ? L"红" : L"黑", x + scale(16), rects.chips[sideIndex][0].top,
                   scale(34), rects.chips[sideIndex][0].bottom - rects.chips[sideIndex][0].top,
                   g_fontUi, sideIndex == 0 ? kRedSoft : kMuted,
                   DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        for (int n = 1; n <= 9; ++n) {
            RECT chip = rects.chips[sideIndex][n - 1];
            const int remain = g_position.remaining(player, n);
            const bool selected = rowActive && humanToMove() && n == g_selected;
            roundRect(dc, chip, scale(6), remain > 0 ? kPanelSoft : kPanelDeep,
                      selected ? kGold : RGB(58, 64, 76), selected ? 2 : 1);
            drawTextAt(dc, std::to_wstring(n), chip.left, chip.top + scale(1),
                       chip.right - chip.left, scale(25), g_fontUi,
                       remain > 0 ? (sideIndex == 0 ? RGB(240, 179, 163) : RGB(215, 220, 230))
                                  : RGB(94, 99, 109));
            for (int dot = 0; dot < 3; ++dot) {
                const int cx = chip.left + (chip.right - chip.left) / 2 + (dot - 1) * scale(8);
                const int cy = chip.bottom - scale(9);
                filledCircle(dc, cx, cy, scale(3), dot < remain ? (sideIndex == 0 ? kRed : kMuted)
                                                               : RGB(63, 69, 80));
            }
        }
    }

    const Player humanTurn = g_position.sideToMove();
    const int selectedRemain = g_position.remaining(humanTurn, g_selected);
    std::wstring selection = L"当前选择：";
    selection += std::to_wstring(g_selected);
    selection += L"（剩 ";
    selection += std::to_wstring(selectedRemain);
    selection += L" 枚）";
    drawTextAt(dc, selection, x + scale(16), rects.mode.top - scale(30), w - scale(32), scale(24),
               g_fontSmall, kGold, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    drawTextAt(dc, modeLabel(), rects.mode.left, rects.mode.top, rects.mode.right - rects.mode.left,
               rects.mode.bottom - rects.mode.top, g_fontSmall, kMuted,
               DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    drawEngineAnalysis(dc, rects);

    roundRect(dc, rects.newButton, scale(8), RGB(98, 77, 42), kGold);
    drawTextAt(dc, L"新局  F2", rects.newButton.left, rects.newButton.top,
               rects.newButton.right - rects.newButton.left,
               rects.newButton.bottom - rects.newButton.top, g_fontUi, kText);
    roundRect(dc, rects.undoButton, scale(8), kPanelSoft, RGB(68, 75, 88));
    drawTextAt(dc, L"悔棋  Ctrl+Z", rects.undoButton.left, rects.undoButton.top,
               rects.undoButton.right - rects.undoButton.left,
               rects.undoButton.bottom - rects.undoButton.top, g_fontUi, kText);
}

void drawStatus(HDC dc, const Layout& layout) {
    const int y = layout.client.bottom - scale(34);
    line(dc, 0, y, layout.client.right, y, RGB(48, 54, 65));
    std::wstring left = L"数字键 1–9 选子 · 点击棋盘落子 · F2 新局 · Ctrl+Z 悔棋";
    if (!g_position.gameOver() && g_hoverCell >= 0 && g_engineAnalysis.valid &&
        g_engineAnalysis.hasCell[g_hoverCell]) {
        const auto& info = g_engineAnalysis.cells[g_hoverCell];
        // Same convention as the marks: exact value when solved, Wilson lower
        // bound otherwise, so a 1-visit sample never reads as a certainty.
        const double shown = info.proven ? info.winrate : winRateLowerBound(info.winrate, info.visits);
        left = L"格 " + widen(squareName(g_hoverCell)) + L" · 数字 " +
               std::to_wstring(static_cast<int>(info.move.number)) + L" · 胜率 " +
               std::to_wstring(static_cast<int>(std::lround(shown * 100.0))) + L"%";
        // A solved cell has no sample to count: the exact endgame solver answers
        // without running a simulation, so "访问 0" would read as an empty search.
        if (info.visits > 0) left += L" · 访问 " + std::to_wstring(info.visits);
        if (info.hasScore) {
            // A solved cell's number is the exact margin; anything else is an
            // average over the visits that carried a score.
            left += info.proven ? L" · 目差 " : L" · 均分 ";
            left += signedDecimal(info.averageScore);
        } else {
            left += L" · 均分 --";
        }
        if (info.proven) left += L"（已解算）";
    } else if (g_thinking) {
        left = L"电脑思考中，请稍候…";
    } else if (g_position.gameOver()) {
        const Outcome outcome = g_position.outcome(g_aValue);
        left = std::wstring(L"终局：") + outcomeText(outcome) + L" · 比分 " +
               signedText(g_position.rawScore()) + L" + A(" + signedValue(g_aValue) + L") = " +
               signedText(g_position.rawScore() + g_aValue);
    }
    drawTextAt(dc, left, scale(24), y, layout.client.right - scale(180), scale(33), g_fontSmall,
               g_position.gameOver() ? kGold : kMuted, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    const std::wstring right = L"第 " + std::to_wstring(g_position.ply()) + L" / 54 手";
    drawTextAt(dc, right, layout.client.right - scale(160), y, scale(136), scale(33), g_fontSmall,
               kMuted, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
}

void drawOverlay(HDC dc, const Layout& layout) {
    if (!g_position.gameOver()) return;
    RECT box = overlayRect(layout);
    roundRect(dc, box, scale(16), RGB(31, 36, 45), kGold, scale(2));
    const Outcome outcome = g_position.outcome(g_aValue);
    const COLORREF color = outcome == Outcome::RedWin ? RGB(240, 137, 122)
                          : outcome == Outcome::BlackWin ? RGB(196, 202, 214)
                                                         : kGreen;
    drawTextAt(dc, outcomeText(outcome), box.left + scale(20), box.top + scale(16),
               box.right - box.left - scale(40), scale(58), g_fontTitle, color);
    const std::wstring detail = L"局面分 " + signedText(g_position.rawScore()) + L"   +   A值 " +
                                signedValue(g_aValue) + L"   =   " +
                                signedText(g_position.rawScore() + g_aValue);
    drawTextAt(dc, detail, box.left + scale(16), box.top + scale(76), box.right - box.left - scale(32),
               scale(28), g_fontUi, kText);
    RECT button = overlayButton(layout);
    roundRect(dc, button, scale(8), RGB(98, 77, 42), kGold);
    drawTextAt(dc, L"再来一局  F2", button.left, button.top, button.right - button.left,
               button.bottom - button.top, g_fontUi, kText);
}

void paintAll(HDC dc, const RECT& client) {
    fillRectColor(dc, client, kBackground);
    Layout layout = computeLayout(g_hwnd);
    drawBoard(dc, layout);
    drawPanel(dc, layout);
    drawStatus(dc, layout);
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

void maybeAiMove();

void resetGame() {
    g_position.reset();
    g_position.setRuleset(g_ruleset);
    g_selected = 1;
    g_hoverCell = -1;
    g_thinking = false;
    g_enginePending = false;
    clearEngineAnalysis();
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
    g_aValue = value;
    if (g_engineChoice != EngineChoice::Off && g_engineReady) {
        sendEngine("set_a " + formatAValue(g_aValue));
        syncEnginePosition();
        if (aiToMove()) maybeAiMove();
    }
    refresh();
}

void undoMove() {
    if (g_position.history().empty()) return;
    const int plies = g_mode == Mode::Pvp ? 1 : 2;
    for (int i = 0; i < plies && !g_position.history().empty(); ++i) g_position.undo();
    g_selected = firstAvailable(g_position.sideToMove());
    if (g_engineChoice != EngineChoice::Off && g_engineReady) syncEnginePosition();
    if (aiToMove()) maybeAiMove();
    else refresh(true);
}

void selectNumber(int number) {
    if (!humanToMove() || g_thinking) return;
    if (number < 1 || number > 9) return;
    if (g_position.remaining(g_position.sideToMove(), number) <= 0) return;
    g_selected = number;
    refresh();
}

// The window contains no search.  When it is the engine's turn and none is
// running, say so once instead of playing a move nobody asked for.
void needEngine(const wchar_t* detail) {
    g_thinking = false;
    g_engineMessage = detail ? detail : L"没有可用的引擎：引擎菜单 → 选择引擎程序…";
    refresh();
}

void maybeAiMove() {
    if (!aiToMove() || g_thinking || g_enginePending) return;
    if (g_engineChoice != EngineChoice::Off && g_engineReady && g_engineClient.running()) {
        g_enginePending = true;
        g_engineMessage = L"引擎思考中…";
        clearEngineAnalysis();
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
    if (!g_position.play(move)) {
        MessageBeep(MB_ICONWARNING);
        return;
    }
    notifyEnginePlay(move, mover);
    settleForcedPasses();
    if (humanToMove()) g_selected = firstAvailable(g_position.sideToMove());
    refresh(true);
    if (aiToMove()) maybeAiMove();
}

void handleClick(POINT point) {
    Layout layout = computeLayout(g_hwnd);
    PanelRects panel = computePanelRects(layout);

    if (g_position.gameOver()) {
        RECT button = overlayButton(layout);
        if (contains(button, point)) resetGame();
        return;
    }

    if (contains(panel.newButton, point)) {
        resetGame();
        return;
    }
    if (contains(panel.undoButton, point)) {
        undoMove();
        return;
    }
    if (contains(panel.aMinus, point)) {
        setAValue(g_aValue - 1);
        return;
    }
    if (contains(panel.aPlus, point)) {
        setAValue(g_aValue + 1);
        return;
    }
    if (contains(panel.mode, point)) {
        if (g_mode == Mode::Pvp) setMode(Mode::HumanRed);
        else if (g_mode == Mode::HumanRed) setMode(Mode::HumanBlack);
        else setMode(Mode::Pvp);
        return;
    }
    if (contains(panel.analysis, point)) {
        toggleAnalysis();
        return;
    }
    for (int side = 0; side < 2; ++side) {
        if (side != static_cast<int>(g_position.sideToMove())) continue;
        for (int n = 0; n < 9; ++n) {
            if (contains(panel.chips[side][n], point)) {
                selectNumber(n + 1);
                return;
            }
        }
    }

    if (contains(layout.board, point)) {
        const int x = (point.x - layout.board.left) / layout.cell;
        const int row = (point.y - layout.board.top) / layout.cell;
        if (x >= 0 && x < 9 && row >= 0 && row < 9) {
            handleBoardClick(index(x, 8 - row));
        }
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
    g_fontTitle = makeFont(30, true);
    g_fontSubtitle = makeFont(10, false);
    g_fontTurn = makeFont(16, true);
    g_fontBig = makeFont(20, true);
    g_fontPiece = makeFont(20, true);
    g_fontUi = makeFont(13, false);
    g_fontSmall = makeFont(11, false);
    g_fontCoord = makeFont(11, false);
}

void destroyFonts() {
    for (HFONT* font : {&g_fontTitle, &g_fontSubtitle, &g_fontTurn, &g_fontBig, &g_fontPiece,
                        &g_fontUi, &g_fontSmall, &g_fontCoord}) {
        if (*font) DeleteObject(*font);
        *font = nullptr;
    }
}

void createMenuBar(HWND hwnd) {
    HMENU menu = CreateMenu();
    HMENU game = CreatePopupMenu();
    HMENU mode = CreatePopupMenu();
    HMENU engine = CreatePopupMenu();
    HMENU rulesetMenu = CreatePopupMenu();
    HMENU aValue = CreatePopupMenu();
    HMENU help = CreatePopupMenu();

    HMENU model = CreatePopupMenu();

    AppendMenuW(game, MF_STRING, ID_NEW, L"新局\tF2");
    AppendMenuW(game, MF_STRING, ID_UNDO, L"悔棋\tCtrl+Z");
    AppendMenuW(game, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(game, MF_STRING, ID_EXIT, L"退出");

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
    AppendMenuW(engine, MF_STRING, ID_ENGINE_PRUNING, L"剪枝（Alpha-Beta）");
    AppendMenuW(engine, MF_STRING, ID_ENGINE_MCTS, L"MCTS（无网络）");
    // One entry per discovered model, filled in by populateModelMenu().
    AppendMenuW(engine, MF_POPUP | MF_STRING, reinterpret_cast<UINT_PTR>(model), L"MCTS + 网络(&N)");
    g_modelMenuPosition = GetMenuItemCount(engine) - 1;
    AppendMenuW(engine, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(engine, MF_STRING, ID_ENGINE_PICK, L"选择引擎程序…");
    AppendMenuW(engine, MF_STRING, ID_ENGINE_AUTO, L"自动探测引擎");
    AppendMenuW(engine, MF_STRING, ID_ENGINE_INFO, L"引擎信息…");
    AppendMenuW(engine, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(engine, MF_STRING, ID_ANALYZE, L"分析当前局面（空格）\tF3");
    AppendMenuW(engine, MF_STRING, ID_ENGINE_STOP, L"停止引擎搜索");
    AppendMenuW(engine, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(engine, MF_STRING, ID_SETTINGS, L"设置（搜索量 / A值）…");

    AppendMenuW(aValue, MF_STRING, ID_A_PLUS, L"A值 +1\t]");
    AppendMenuW(aValue, MF_STRING, ID_A_MINUS, L"A值 −1\t[");
    AppendMenuW(aValue, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(aValue, MF_STRING, ID_A_ZERO, L"设为 0");
    AppendMenuW(aValue, MF_STRING, ID_A_ONE, L"设为 +1（当前规则）");
    AppendMenuW(aValue, MF_STRING, ID_A_TWO, L"设为 +2");

    AppendMenuW(help, MF_STRING, ID_RULES, L"规则说明");
    AppendMenuW(help, MF_STRING, ID_ABOUT, L"关于数弈");

    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(game), L"游戏(&G)");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(mode), L"模式(&M)");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(rulesetMenu), L"规则(&R)");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(engine), L"引擎(&E)");
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(aValue), L"A值(&A)");
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

void showRules() {
    const wchar_t* text =
        L"数弈 · 规则\n\n"
        L"1. 9×9 棋盘，横坐标 a–i，纵坐标 1–9。计算距离时 a–i 按 1–9 计。\n"
        L"2. 红黑双方，红先，轮流在空格落子；棋子落下后不可移动、不可移除。\n"
        L"3. 每方有数字 1–9 各三枚，共 27 枚。同数字棋子彼此完全相同。\n"
        L"4. 双方 54 枚棋子全部落完后终局。\n\n"
        L"计分：\n"
        L"· 单格分 = sgn(附近红子最大数字 − 附近黑子最大数字)。\n"
        L"· “附近”指曼哈顿距离 ≤1 的格子，包含该格自身；棋盘边界不回绕。\n"
        L"· 没有某种颜色的棋子时，该颜色部分按 0 计。\n"
        L"· 局面分 = 81 个格子分之和 + A值。\n"
        L"· 局面分 >0 红胜，<0 黑胜，=0 平局。当前 A值 = +1。\n\n"
        L"操作：数字键 1–9 选择数字；点击棋盘落子；F2 新局；Ctrl+Z 悔棋；\n"
        L"A值菜单或 [ / ] 可调节 A值。";
    MessageBoxW(g_hwnd, text, L"数弈规则", MB_OK | MB_ICONINFORMATION);
}

void showAbout() {
    const wchar_t* text =
        L"数弈 1.0\n\n"
        L"一个 9×9 的数字落子棋，原生 Win32/GDI 界面，非 Web 实现。\n"
        L"C++17 引擎进程：剪枝、MCTS、MCTS+NN；GTP-like 异步协议。\n\n"
        L"当前规则版本：A值 = +1。\n"
        L"ELO 与 A值分析见 docs/。";
    MessageBoxW(g_hwnd, text, L"关于数弈", MB_OK | MB_ICONINFORMATION);
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
        return 0;

    case WM_TIMER:
        if (wParam == 1) {
            drainEngineLines();
            if (g_engineChoice != EngineChoice::Off && (g_analysisOn || g_enginePending)) {
                refresh();
            }
        }
        return 0;

    case WM_GETMINMAXINFO: {
        auto* info = reinterpret_cast<MINMAXINFO*>(lParam);
        info->ptMinTrackSize.x = scale(940);
        info->ptMinTrackSize.y = scale(700);
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

    case WM_KEYDOWN: {
        const bool control = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
        if (wParam == VK_F2) {
            resetGame();
            return 0;
        }
        // Space is the analysis switch: start, stop (keeping the numbers) and
        // continue past the visit limit.  F3 stays as an alias.
        if (wParam == VK_SPACE || wParam == VK_F3) {
            toggleAnalysis();
            return 0;
        }
        if (control && (wParam == 'Z' || wParam == 'z')) {
            undoMove();
            return 0;
        }
        if (wParam >= '1' && wParam <= '9') {
            selectNumber(static_cast<int>(wParam - '0'));
            return 0;
        }
        if (wParam >= VK_NUMPAD1 && wParam <= VK_NUMPAD9) {
            selectNumber(static_cast<int>(wParam - VK_NUMPAD0));
            return 0;
        }
        if (wParam == VK_OEM_4) {
            setAValue(g_aValue - 1);
            return 0;
        }
        if (wParam == VK_OEM_6) {
            setAValue(g_aValue + 1);
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
        case ID_NEW: resetGame(); return 0;
        case ID_UNDO: undoMove(); return 0;
        case ID_EXIT: DestroyWindow(hwnd); return 0;
        case ID_MODE_PVP: setMode(Mode::Pvp); return 0;
        case ID_MODE_HUMAN_RED: setMode(Mode::HumanRed); return 0;
        case ID_MODE_HUMAN_BLACK: setMode(Mode::HumanBlack); return 0;
        case ID_RULESET_CLASSIC: setRuleset(Ruleset::Classic); return 0;
        case ID_RULESET_BLOCKING: setRuleset(Ruleset::Blocking); return 0;
        case ID_RULESET_NOPASS: setRuleset(Ruleset::NoPass); return 0;
        case ID_ENGINE_OFF: setEngineChoice(EngineChoice::Off); return 0;
        case ID_ENGINE_PRUNING: setEngineChoice(EngineChoice::Pruning); return 0;
        case ID_ENGINE_MCTS: setEngineChoice(EngineChoice::Mcts); return 0;
        case ID_ENGINE_MCTS_NN: setEngineChoice(EngineChoice::MctsNn); return 0;
        case ID_ENGINE_PICK: pickEngineProgram(); return 0;
        case ID_ENGINE_AUTO: useAutoEngine(); return 0;
        case ID_ENGINE_INFO: showEngineInfo(); return 0;
        case ID_MODEL_RESCAN: {
            scanModels();
            const int index = modelIndexFor(g_nnModelPath);
            g_nnModelPath = index >= 0 ? g_nnModelPath : std::string();
            populateModelMenu();
            syncModelChecks();
            refresh();
            return 0;
        }
        case ID_ANALYZE: toggleAnalysis(); return 0;
        case ID_SETTINGS: openSettings(); return 0;
        case ID_ENGINE_STOP: {
            const bool wasPending = g_enginePending;
            g_enginePending = false;
            g_pendingReplyId.clear();
            g_analysisOn = false;
            if (g_engineClient.running()) sendEngine("stop");
            clearEngineAnalysis();
            g_engineMessage = wasPending ? L"已停止思考：轮到的这手需要重新让引擎走（引擎菜单）"
                                         : L"引擎已停止";
            g_thinking = false;
            refresh();
            return 0;
        }
        case ID_A_PLUS: setAValue(g_aValue + 1); return 0;
        case ID_A_MINUS: setAValue(g_aValue - 1); return 0;
        case ID_A_ZERO: setAValue(0); return 0;
        case ID_A_ONE: setAValue(1); return 0;
        case ID_A_TWO: setAValue(2); return 0;
        case ID_RULES: showRules(); return 0;
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
    // Diagnostics first: the raw exe directory, then the resolved candidates.
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
        while (!tail.empty() && (tail.front() == L' ' || tail.front() == L'	')) tail.erase(0, 1);
        std::string out = "engine_choice.txt";
        if (!tail.empty()) {
            const int needed = WideCharToMultiByte(CP_UTF8, 0, tail.c_str(), -1, nullptr, 0, nullptr, nullptr);
            if (needed > 1) {
                out.assign(static_cast<std::size_t>(needed - 1), '\0');
                WideCharToMultiByte(CP_UTF8, 0, tail.c_str(), -1, out.data(), needed, nullptr, nullptr);
            }
        }
        return dumpEngineChoice(out) ? 0 : 1;
    }
    if (commandLine && std::wstring(commandLine).find(L"--dump-models") != std::wstring::npos) {
        std::wstring argument = commandLine;
        const std::size_t flag = argument.find(L"--dump-models");
        std::wstring tail = argument.substr(flag + std::wstring(L"--dump-models").size());
        while (!tail.empty() && (tail.front() == L' ' || tail.front() == L'\t')) tail.erase(0, 1);
        // The command line is UTF-16; the path may contain Chinese characters, so
        // convert it properly instead of narrowing code unit by code unit.
        std::string dumpPath = "ui_models.txt";
        if (!tail.empty()) {
            const int needed = WideCharToMultiByte(CP_UTF8, 0, tail.c_str(), -1, nullptr, 0,
                                                   nullptr, nullptr);
            if (needed > 1) {
                dumpPath.assign(static_cast<std::size_t>(needed - 1), '\0');
                WideCharToMultiByte(CP_UTF8, 0, tail.c_str(), -1, dumpPath.data(), needed, nullptr,
                                    nullptr);
            }
        }
        const bool ok = dumpModelMenu(dumpPath);
        return ok ? 0 : 1;
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

    RECT windowRect{0, 0, scale(1120), scale(820)};
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

    // If an engine is discoverable next to the window, load it now so analysis is
    // one keypress away.  If there is none, the window still starts: the 引擎 menu
    // is where an engine gets chosen, and nothing pretends to compute.
    if (!chooseEnginePath().empty()) {
        g_engineChoice = EngineChoice::MctsNn;
        if (!startEngineProcess()) {
            g_engineChoice = EngineChoice::Off;
            g_engineReady = false;
        }
        syncEngineChecks();
    }

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return static_cast<int>(message.wParam);
}
