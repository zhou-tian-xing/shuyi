#include "game_record.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>

namespace shuyi {
namespace {

std::string trim(const std::string& text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

// A number with a fixed number of decimals.  The text form has to be stable:
// saving a record that was just loaded must not rewrite it.
std::string fixed(double value, int places) {
    if (value == 0.0) value = 0.0;  // no "-0.0000"
    char buffer[64]{};
    std::snprintf(buffer, sizeof(buffer), "%.*f", places, value);
    return buffer;
}

std::vector<std::string> splitWords(const std::string& text) {
    std::vector<std::string> words;
    std::string current;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) words.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) words.push_back(current);
    return words;
}

bool parseDouble(const std::string& text, double* value) {
    if (text.empty()) return false;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size()) return false;
    if (value) *value = parsed;
    return true;
}

bool parseInteger(const std::string& text, long long* value) {
    if (text.empty()) return false;
    char* end = nullptr;
    const long long parsed = std::strtoll(text.c_str(), &end, 10);
    if (end != text.c_str() + text.size()) return false;
    if (value) *value = parsed;
    return true;
}

} // namespace

std::wstring moveLabel(Move move) {
    if (isPass(move)) return L"pass";
    if (!move.valid()) return std::wstring();
    const std::string text = squareName(move.cell);
    std::wstring wide(text.begin(), text.end());
    wide += L"-";
    wide += std::to_wstring(static_cast<int>(move.number));
    return wide;
}

std::string moveProtocolText(Move move) {
    if (isPass(move)) return "pass";
    if (!move.valid()) return std::string();
    return squareName(move.cell) + ":" + std::to_string(static_cast<int>(move.number));
}

bool parseMoveLabel(const std::string& text, Move& move) {
    if (text == "pass") {
        move = Move(kPassCell, kPassNumber);
        return true;
    }
    // "a4-2" in the record, "a4:2" in the protocol.
    std::string cell;
    std::string number;
    std::size_t split = 0;
    while (split < text.size() && text[split] != '-' && text[split] != ':') ++split;
    cell = text.substr(0, split);
    if (split < text.size()) number = text.substr(split + 1);
    int index = 0;
    if (!parseSquare(cell, index) || number.empty()) return false;
    int count = 0;
    for (char c : number) {
        if (c < '0' || c > '9') return false;
        count = count * 10 + (c - '0');
    }
    if (count < 1 || count > 9) return false;
    move = Move(index, count);
    return true;
}

GameRecord::GameRecord() {
    nodes_.emplace_back();
}

int GameRecord::depth(int id) const {
    int depth = 0;
    for (int walk = parentOf(id); walk >= 0; walk = parentOf(walk)) ++depth;
    return depth;
}

std::vector<int> GameRecord::line(int id) const {
    std::vector<int> path;
    for (int walk = id; walk > 0 && contains(walk); walk = parentOf(walk)) path.push_back(walk);
    std::reverse(path.begin(), path.end());
    return path;
}

int GameRecord::addChild(int parent, Move move) {
    if (!contains(parent)) parent = 0;
    for (int child : nodes_[static_cast<std::size_t>(parent)].children) {
        if (nodes_[static_cast<std::size_t>(child)].move == move) return child;
    }
    const int id = count();
    nodes_.emplace_back();
    RecordNode& node = nodes_.back();
    node.parent = parent;
    node.move = move;
    nodes_[static_cast<std::size_t>(parent)].children.push_back(id);
    return id;
}

int GameRecord::play(int parent, Move move) {
    current_ = addChild(parent, move);
    return current_;
}

int GameRecord::select(int id) {
    if (contains(id)) current_ = id;
    return current_;
}

int GameRecord::up() {
    const int parent = parentOf(current_);
    if (parent < 0) return -1;
    current_ = parent;
    return current_;
}

int GameRecord::down() {
    const auto& children = childrenOf(current_);
    if (children.empty()) return -1;
    current_ = children.front();  // the trunk: leftmost column in the tree view
    return current_;
}

bool GameRecord::erase(int id) {
    if (id <= 0 || !contains(id)) return false;
    const int parent = parentOf(id);
    // Rebuild the tree in pre-order, skipping the deleted subtree.  Rebuilding
    // beats splicing indices: every node id that survives keeps its analysis and
    // there is exactly one place where ids are remapped.
    std::vector<RecordNode> fresh(1);
    std::vector<int> remap(nodes_.size(), -1);
    remap[0] = 0;
    const std::function<int(int, int)> copy = [&](int old, int newParent) -> int {
        const int created = static_cast<int>(fresh.size());
        fresh.emplace_back();
        remap[static_cast<std::size_t>(old)] = created;
        RecordNode& target = fresh.back();
        target.parent = newParent;
        target.move = nodes_[static_cast<std::size_t>(old)].move;
        target.analysis = nodes_[static_cast<std::size_t>(old)].analysis;
        target.run = nodes_[static_cast<std::size_t>(old)].run;
        for (int child : nodes_[static_cast<std::size_t>(old)].children) {
            if (child == id) continue;
            const int copied = copy(child, created);
            fresh[static_cast<std::size_t>(created)].children.push_back(copied);
        }
        return created;
    };
    for (int child : nodes_[0].children) {
        if (child == id) continue;
        // Build the child first: `fresh` reallocates inside copy(), and a
        // reference taken from it before the call would dangle.
        const int copied = copy(child, 0);
        fresh[0].children.push_back(copied);
    }
    nodes_ = std::move(fresh);
    current_ = remap[static_cast<std::size_t>(parent)];
    if (current_ < 0) current_ = 0;
    return true;
}

void GameRecord::setAnalysis(int id, EngineAnalysisView analysis, int run) {
    if (!contains(id)) return;
    RecordNode& node = nodes_[static_cast<std::size_t>(id)];
    node.analysis = std::move(analysis);
    node.run = run;
}

void GameRecord::writeLineText(std::string& out, int node) const {
    const auto& children = childrenOf(node);
    if (children.empty()) return;
    out += moveProtocolText(nodes_[static_cast<std::size_t>(children.front())].move);
    out += ' ';
    for (std::size_t i = 1; i < children.size(); ++i) {
        out += '(';
        out += moveProtocolText(nodes_[static_cast<std::size_t>(children[i])].move);
        out += ' ';
        writeLineText(out, children[i]);
        out += ')';
        out += ' ';
    }
    writeLineText(out, children.front());
}

// The file format, in full:
//
//   # shuyi record v2
//   ruleset blocking
//   moves a4:2 b2:5 (c3:1 ) d4:7
//   #node <path> wr <f> visits <i> proven <0|1> nodes <i> ms <i> best <move|-> score <f|->
//   #cell <path> <move> <wr> <visits> <proven> <has_score> <score|->
//   #cand <path> <move> <wr> <visits> <score>
//
// `<path>` is the move sequence that reaches the node, in the same spelling the
// `moves` line uses, joined by '/' (`.` is the empty board).  A path rather than a
// node number: numbers are an implementation detail of the tree and change when a
// branch is deleted.
std::string GameRecord::save(Ruleset ruleset) const {
    std::string out = "# shuyi record v2\n";
    out += "ruleset ";
    out += rulesetName(ruleset);
    out += '\n';
    out += "moves ";
    writeLineText(out, 0);
    while (!out.empty() && (out.back() == ' ' || out.back() == '\n')) out.pop_back();
    out += '\n';
    writeAnalysis(out, 0, ".");
    return out;
}

void GameRecord::writeAnalysis(std::string& out, int node, const std::string& path) const {
    const EngineAnalysisView& view = nodes_[static_cast<std::size_t>(node)].analysis;
    if (view.valid) {
        out += "#node ";
        out += path;
        out += " wr ";
        out += fixed(view.winrate, 4);
        out += " visits ";
        out += std::to_string(view.visits);
        out += " proven ";
        out += view.proven ? "1" : "0";
        out += " nodes ";
        out += std::to_string(view.nodes);
        out += " ms ";
        out += std::to_string(view.elapsedMs);
        out += " best ";
        out += view.best.valid() ? moveProtocolText(view.best) : std::string("-");
        out += " score ";
        out += view.hasScore ? fixed(view.averageScore, 4) : std::string("-");
        if (!view.backend.empty()) {
            out += " backend ";
            out += view.backend;
        }
        out += '\n';
        for (int cell = 0; cell < kCells; ++cell) {
            if (!view.hasCell[cell]) continue;
            const CellAnalysisView& info = view.cells[cell];
            out += "#cell ";
            out += path;
            out += ' ';
            out += moveProtocolText(info.move);
            out += ' ';
            out += fixed(info.winrate, 4);
            out += ' ';
            out += std::to_string(info.visits);
            out += ' ';
            out += info.proven ? "1" : "0";
            out += ' ';
            out += info.hasScore ? "1" : "0";
            out += ' ';
            out += info.hasScore ? fixed(info.averageScore, 4) : std::string("-");
            out += '\n';
        }
        for (const AnalysisCandidateView& candidate : view.candidates) {
            if (!candidate.move.valid()) continue;
            out += "#cand ";
            out += path;
            out += ' ';
            out += moveProtocolText(candidate.move);
            out += ' ';
            out += fixed(candidate.winrate, 4);
            out += ' ';
            out += std::to_string(candidate.visits);
            out += ' ';
            out += fixed(candidate.averageScore, 4);
            out += '\n';
        }
    }
    for (int child : nodes_[static_cast<std::size_t>(node)].children) {
        const std::string move = moveProtocolText(nodes_[static_cast<std::size_t>(child)].move);
        writeAnalysis(out, child, path == "." ? move : path + "/" + move);
    }
}

bool GameRecord::load(const std::string& text, Ruleset* ruleset, std::string* error) {
    const auto fail = [&](const std::string& why) {
        if (error) *error = why;
        return false;
    };
    Ruleset parsedRuleset = ruleset ? *ruleset : kDefaultRuleset;
    std::string moves;
    std::vector<std::string> analysisLines;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string trimmed = trim(line);
        if (trimmed.empty()) continue;
        // Analysis lines are comments to a reader that does not know them, which is
        // what keeps a record written by this version readable by an older one.
        if (trimmed.rfind("#node ", 0) == 0 || trimmed.rfind("#cell ", 0) == 0 ||
            trimmed.rfind("#cand ", 0) == 0) {
            analysisLines.push_back(trimmed);
            continue;
        }
        if (trimmed[0] == '#') continue;
        if (trimmed.rfind("ruleset ", 0) == 0) {
            if (!parseRuleset(trim(trimmed.substr(8)), parsedRuleset)) {
                return fail("unknown rule set: " + trimmed.substr(8));
            }
            continue;
        }
        if (trimmed.rfind("moves ", 0) == 0) {
            moves += trimmed.substr(6);
            moves += ' ';
            continue;
        }
        // A bare move list (no header) is accepted: it is what a user is likely
        // to paste or write by hand.
        moves += trimmed;
        moves += ' ';
    }
    if (trim(moves).empty()) return fail("the record has no moves");

    // Tokens: parentheses are their own tokens even without spaces around them.
    std::vector<std::string> tokens;
    std::string current;
    for (char c : moves) {
        if (c == '(' || c == ')') {
            if (!current.empty()) tokens.push_back(current);
            current.clear();
            tokens.emplace_back(1, c);
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) tokens.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) tokens.push_back(current);

    nodes_.clear();
    nodes_.emplace_back();
    current_ = 0;
    // PGN variation semantics: `(` starts an alternative to the move just played,
    // so the variation's parent is that move's parent, and `)` returns to the
    // move being replaced.
    struct Saved {
        int parent = 0;
        int last = -1;
    };
    std::vector<Saved> stack;
    int parent = 0;
    int last = -1;
    int mainTail = 0;
    for (const std::string& token : tokens) {
        if (token == "(") {
            stack.push_back({parent, last});
            parent = last >= 0 ? parentOf(last) : parent;
            last = -1;
            continue;
        }
        if (token == ")") {
            if (stack.empty()) return fail("unbalanced ')' in the record");
            parent = stack.back().parent;
            last = stack.back().last;
            stack.pop_back();
            continue;
        }
        Move move;
        if (!parseMoveLabel(token, move)) return fail("not a move: " + token);
        parent = addChild(parent, move);
        last = parent;
        if (stack.empty()) mainTail = parent;
    }
    if (!stack.empty()) return fail("unbalanced '(' in the record");
    current_ = mainTail;
    for (const std::string& analysisLine : analysisLines) {
        if (!loadAnalysisLine(analysisLine, error)) return false;
    }
    if (ruleset) *ruleset = parsedRuleset;
    if (error) error->clear();
    return true;
}

namespace {

// The node a `<path>` names, or -1 when the record has no such line.
int nodeForPath(const std::vector<RecordNode>& nodes, const std::string& path) {
    if (path == ".") return 0;
    int node = 0;
    std::size_t start = 0;
    while (true) {
        const std::size_t slash = path.find('/', start);
        const std::string token =
            path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        Move move;
        if (!parseMoveLabel(token, move)) return -1;
        int found = -1;
        for (int child : nodes[static_cast<std::size_t>(node)].children) {
            if (nodes[static_cast<std::size_t>(child)].move == move) {
                found = child;
                break;
            }
        }
        if (found < 0) return -1;
        node = found;
        if (slash == std::string::npos) return node;
        start = slash + 1;
    }
}

} // namespace

bool GameRecord::loadAnalysisLine(const std::string& line, std::string* error) {
    const auto fail = [&](const std::string& why) {
        if (error) *error = why;
        return false;
    };
    const std::vector<std::string> word = splitWords(line);
    if (word.size() < 2) return fail("truncated analysis line: " + line);
    const int node = nodeForPath(nodes_, word[1]);
    if (node < 0) return fail("analysis for a move that is not in the record: " + word[1]);
    EngineAnalysisView& view = nodes_[static_cast<std::size_t>(node)].analysis;
    if (word[0] == "#cell") {
        // #cell <path> <move> <wr> <visits> <proven> <has_score> <score>
        if (word.size() < 8) return fail("truncated cell line: " + line);
        Move move;
        if (!parseMoveLabel(word[2], move) || isPass(move)) {
            return fail("not a placement: " + word[2]);
        }
        CellAnalysisView info;
        info.move = move;
        info.hasData = true;
        if (!parseDouble(word[3], &info.winrate)) return fail("bad cell win rate: " + word[3]);
        long long visits = 0;
        if (!parseInteger(word[4], &visits)) return fail("bad cell visits: " + word[4]);
        info.visits = static_cast<int>(std::clamp<long long>(visits, 0, 1LL << 30));
        info.proven = word[5] == "1";
        info.hasScore = word[6] == "1";
        if (info.hasScore && !parseDouble(word[7], &info.averageScore)) {
            return fail("bad cell score: " + word[7]);
        }
        view.cells[move.cell] = info;
        view.hasCell[move.cell] = true;
        view.valid = true;
        return true;
    }
    if (word[0] == "#cand") {
        // #cand <path> <move> <wr> <visits> <score>
        if (word.size() < 6) return fail("truncated candidate line: " + line);
        Move move;
        if (!parseMoveLabel(word[2], move)) return fail("not a move: " + word[2]);
        AnalysisCandidateView candidate;
        candidate.move = move;
        if (!parseDouble(word[3], &candidate.winrate)) return fail("bad candidate win rate");
        long long visits = 0;
        if (!parseInteger(word[4], &visits)) return fail("bad candidate visits");
        candidate.visits = static_cast<int>(std::clamp<long long>(visits, 0, 1LL << 30));
        if (!parseDouble(word[5], &candidate.averageScore)) return fail("bad candidate score");
        view.candidates.push_back(candidate);
        view.valid = true;
        return true;
    }
    // #node <path> key value ...
    if (word.size() % 2 != 0) return fail("odd field list in a node line: " + line);
    for (std::size_t i = 2; i + 1 < word.size(); i += 2) {
        const std::string& key = word[i];
        const std::string& value = word[i + 1];
        if (key == "wr") {
            if (!parseDouble(value, &view.winrate)) return fail("bad win rate: " + value);
        } else if (key == "visits") {
            long long visits = 0;
            if (!parseInteger(value, &visits)) return fail("bad visits: " + value);
            view.visits = static_cast<int>(std::clamp<long long>(visits, 0, 1LL << 30));
        } else if (key == "proven") {
            view.proven = value == "1";
        } else if (key == "nodes") {
            if (!parseInteger(value, &view.nodes)) return fail("bad node count: " + value);
        } else if (key == "ms") {
            if (!parseInteger(value, &view.elapsedMs)) return fail("bad elapsed time: " + value);
        } else if (key == "best") {
            if (value == "-") {
                view.best = Move{};
            } else {
                Move move;
                if (!parseMoveLabel(value, move)) return fail("bad best move: " + value);
                view.best = move;
            }
        } else if (key == "score") {
            if (value == "-") view.hasScore = false;
            else {
                if (!parseDouble(value, &view.averageScore)) return fail("bad score: " + value);
                view.hasScore = true;
            }
        } else if (key == "backend") {
            view.backend = value;
        }
        // An unknown key is skipped: a newer writer may add fields.
    }
    view.valid = true;
    return true;
}

bool moveVerdict(const RecordNode& parent, const RecordNode& child, MoveVerdict* verdict) {
    if (!parent.analysis.valid || !child.analysis.valid) return false;
    // The parent's verdict is from the mover's side and the child's from the
    // opponent's, so the child has to be flipped before they can be subtracted.
    const double actual = 1.0 - child.analysis.winrate;
    bool scored = false;
    double expected = parent.analysis.winrate;
    for (const AnalysisCandidateView& candidate : parent.analysis.candidates) {
        if (candidate.move != child.move) continue;
        expected = candidate.winrate;
        scored = true;
        break;
    }
    if (!scored && !isPass(child.move)) {
        const int cell = child.move.cell;
        if (cell >= 0 && cell < kCells && parent.analysis.hasCell[cell]) {
            expected = parent.analysis.cells[cell].winrate;
            scored = true;
        }
    }
    if (verdict) {
        verdict->delta = actual - expected;
        verdict->scored = scored;
    }
    return true;
}

} // namespace shuyi
