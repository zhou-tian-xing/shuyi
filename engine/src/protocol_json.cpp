#include "shuyi/protocol_json.h"

#include "shuyi/format.h"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace shuyi {
namespace {

// Fixed precision, and never "nan"/"inf": a JSON document with those is not
// valid JSON, and a front end must never have to special-case them.
std::string number(double value, int decimals) {
    if (!std::isfinite(value)) value = 0.0;
    char buffer[64]{};
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return buffer;
}

std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char escape[8]{};
                std::snprintf(escape, sizeof(escape), "\\u%04x", static_cast<unsigned char>(c));
                out += escape;
            } else {
                out += c;
            }
        }
    }
    out += '"';
    return out;
}

std::string moveText(Move move) {
    if (!move.valid()) return "none";
    return squareName(move.cell) + ":" + std::to_string(static_cast<int>(move.number));
}

} // namespace

std::string analysisJson(const Analysis& analysis, const std::string& backend,
                         const std::string& turn, long long elapsedMs, const std::string& nn,
                         bool aInput) {
    std::ostringstream out;
    out << '{';
    out << "\"engine\":" << quoted(engineName(analysis.kind));
    out << ",\"turn\":" << quoted(turn);
    out << ",\"backend\":" << quoted(backend);
    out << ",\"visits\":" << analysis.visits;
    out << ",\"nodes\":" << analysis.nodes;
    out << ",\"depth\":" << analysis.depth;
    out << ",\"proven\":" << (analysis.proven ? "true" : "false");
    out << ",\"winrate\":" << number(analysis.winrate, 4);
    out << ",\"has_score\":" << (analysis.hasScore ? "true" : "false");
    out << ",\"score\":" << number(analysis.averageScore, 3);
    out << ",\"best\":" << quoted(moveText(analysis.best));
    out << ",\"elapsed_ms\":" << elapsedMs;
    out << ",\"nn\":" << quoted(nn);
    out << ",\"a_input\":" << (aInput ? 1 : 0);

    if (!analysis.pv.empty()) {
        out << ",\"pv\":[";
        std::size_t start = 0;
        bool first = true;
        while (start <= analysis.pv.size()) {
            const std::size_t comma = analysis.pv.find(',', start);
            const std::string step = analysis.pv.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!step.empty()) {
                if (!first) out << ',';
                out << quoted(step);
                first = false;
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        out << ']';
    }
    if (analysis.kind == EngineKind::Pruning) {
        out << ",\"window\":" << analysis.window;
        out << ",\"window_researches\":" << analysis.windowResearches;
        out << ",\"pvs_researches\":" << analysis.pvsResearches;
    }

    out << ",\"moves\":[";
    for (std::size_t i = 0; i < analysis.candidates.size(); ++i) {
        const AnalysisCandidate& candidate = analysis.candidates[i];
        if (i) out << ',';
        out << "{\"move\":" << quoted(moveText(candidate.move))
            << ",\"winrate\":" << number(candidate.winrate, 4)
            << ",\"visits\":" << candidate.visits
            << ",\"score\":" << number(candidate.averageScore, 3) << '}';
    }
    out << ']';

    out << ",\"cells\":[";
    bool firstCell = true;
    for (const CellAnalysis& cell : analysis.cells) {
        if (!cell.hasData) continue;
        if (!firstCell) out << ',';
        firstCell = false;
        out << "{\"cell\":" << quoted(squareName(cell.cell))
            << ",\"number\":" << static_cast<int>(cell.move.number)
            << ",\"winrate\":" << number(cell.winrate, 4)
            << ",\"visits\":" << cell.visits
            << ",\"score\":" << number(cell.averageScore, 3)
            << ",\"proven\":" << (cell.proven ? "true" : "false") << '}';
    }
    out << ']';
    out << '}';
    return out.str();
}

} // namespace shuyi
