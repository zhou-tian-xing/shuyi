#include "game_record.h"

#include <algorithm>
#include <cctype>
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

void GameRecord::setAnalysis(int id, EngineAnalysisView analysis) {
    if (!contains(id)) return;
    nodes_[static_cast<std::size_t>(id)].analysis = std::move(analysis);
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

std::string GameRecord::save(Ruleset ruleset) const {
    std::string out = "# shuyi record v1\n";
    out += "ruleset ";
    out += rulesetName(ruleset);
    out += '\n';
    out += "moves ";
    writeLineText(out, 0);
    while (!out.empty() && (out.back() == ' ' || out.back() == '\n')) out.pop_back();
    out += '\n';
    return out;
}

bool GameRecord::load(const std::string& text, Ruleset* ruleset, std::string* error) {
    const auto fail = [&](const std::string& why) {
        if (error) *error = why;
        return false;
    };
    Ruleset parsedRuleset = ruleset ? *ruleset : kDefaultRuleset;
    std::string moves;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
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
    if (ruleset) *ruleset = parsedRuleset;
    if (error) error->clear();
    return true;
}

bool moveDelta(const RecordNode& parent, const RecordNode& child, double* delta) {
    if (!child.analysis.valid) return false;
    // What the parent's analysis said the move was worth, from the mover's side.
    bool havePrediction = false;
    double predicted = 0.0;
    for (const AnalysisCandidateView& candidate : parent.analysis.candidates) {
        if (candidate.move != child.move) continue;
        predicted = candidate.winrate;
        havePrediction = true;
        break;
    }
    if (!havePrediction) {
        const int cell = child.move.cell;
        if (cell < 0 || cell >= kCells || !parent.analysis.hasCell[cell]) return false;
        predicted = displayWinRate(parent.analysis.cells[cell]);
        havePrediction = true;
    }
    if (!havePrediction) return false;
    // The child's numbers are from the opponent's point of view.
    const double actual = 1.0 - child.analysis.winrate;
    if (delta) *delta = actual - predicted;
    return true;
}

} // namespace shuyi
