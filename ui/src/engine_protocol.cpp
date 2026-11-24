#include "engine_view.h"

#include "json.h"

#include <cctype>
#include <cstdlib>

namespace shuyi {
namespace {

int intOf(const JsonValue& object, const char* key, int fallback = 0) {
    const JsonValue* value = object.find(key);
    return value ? value->asInt(fallback) : fallback;
}

double doubleOf(const JsonValue& object, const char* key, double fallback = 0.0) {
    const JsonValue* value = object.find(key);
    return value ? value->asNumber(fallback) : fallback;
}

bool boolOf(const JsonValue& object, const char* key, bool fallback = false) {
    const JsonValue* value = object.find(key);
    return value ? value->asBool(fallback) : fallback;
}

std::string stringOf(const JsonValue& object, const char* key) {
    const JsonValue* value = object.find(key);
    return value && value->isString() ? value->asString() : std::string();
}

} // namespace

Move parseMoveToken(const std::string& text) {
    if (text.empty() || text == "none" || text == "pass") return Move{};
    std::string cell;
    std::string number;
    std::size_t split = 0;
    while (split < text.size() && text[split] != ':' && text[split] != ' ') ++split;
    cell = text.substr(0, split);
    if (split < text.size()) {
        std::size_t start = split + 1;
        while (start < text.size() && text[start] == ' ') ++start;
        number = text.substr(start);
    }
    int index = 0;
    if (!parseSquare(cell, index)) return Move{};
    if (number.empty()) return Move{};
    int count = 0;
    try {
        count = std::stoi(number);
    } catch (...) {
        return Move{};
    }
    if (count < 1 || count > 9) return Move{};
    return Move(index, count);
}

bool parseAnalysis(const std::string& json, EngineAnalysisView& out, std::string* error) {
    JsonValue root;
    if (!parseJson(json, root, error)) return false;
    if (!root.isObject()) {
        if (error) *error = "analysis is not a JSON object";
        return false;
    }

    EngineAnalysisView view;
    view.kind = stringOf(root, "engine");
    view.turn = stringOf(root, "turn");
    view.backend = stringOf(root, "backend");
    view.nn = stringOf(root, "nn");
    view.visits = intOf(root, "visits");
    view.nodes = intOf(root, "nodes");
    view.depth = intOf(root, "depth");
    view.proven = boolOf(root, "proven");
    view.hasScore = boolOf(root, "has_score");
    view.winrate = doubleOf(root, "winrate", 0.5);
    view.averageScore = doubleOf(root, "score");
    view.score = static_cast<int>(view.averageScore < 0 ? view.averageScore - 0.5
                                                        : view.averageScore + 0.5);
    view.elapsedMs = intOf(root, "elapsed_ms");
    view.window = intOf(root, "window");
    view.windowResearches = intOf(root, "window_researches");
    view.pvsResearches = intOf(root, "pvs_researches");
    view.best = parseMoveToken(stringOf(root, "best"));
    view.valid = view.best.valid() || view.kind == "random";

    const JsonValue* pv = root.find("pv");
    if (pv && pv->isArray()) {
        for (const JsonValue& step : pv->items) {
            if (!step.isString()) continue;
            if (!view.pv.empty()) view.pv += ",";
            view.pv += step.asString();
        }
    }

    const JsonValue* moves = root.find("moves");
    if (moves && moves->isArray()) {
        for (const JsonValue& item : moves->items) {
            if (!item.isObject()) continue;
            AnalysisCandidateView candidate;
            candidate.move = parseMoveToken(stringOf(item, "move"));
            if (!candidate.move.valid()) continue;
            candidate.winrate = doubleOf(item, "winrate", 0.5);
            candidate.visits = intOf(item, "visits");
            candidate.averageScore = doubleOf(item, "score");
            view.candidates.push_back(candidate);
        }
    }

    const JsonValue* cells = root.find("cells");
    if (cells && cells->isArray()) {
        for (const JsonValue& item : cells->items) {
            if (!item.isObject()) continue;
            int index = 0;
            if (!parseSquare(stringOf(item, "cell"), index)) continue;
            CellAnalysisView info;
            info.move = Move(index, static_cast<std::uint8_t>(intOf(item, "number")));
            info.winrate = doubleOf(item, "winrate", 0.5);
            info.visits = intOf(item, "visits");
            info.averageScore = doubleOf(item, "score");
            info.hasData = info.visits > 0;
            info.proven = boolOf(item, "proven");
            // Per cell, not per position: a solved move has an exact margin even
            // when the network has no score head, and an unsolved one may have no
            // score at all.  An older engine does not send the field; there the
            // position-level flag was the same decision for every cell, so it is
            // the honest fallback.
            info.hasScore = boolOf(item, "has_score", view.hasScore);
            view.cells[index] = info;
            view.hasCell[index] = true;
        }
    }

    out = std::move(view);
    return true;
}

} // namespace shuyi
