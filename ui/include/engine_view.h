// The analysis a front end displays, and the parser that fills it from the
// engine's JSON.
//
// One object per `info` line.  The view keeps exactly what the window draws:
// the global verdict plus one entry per board cell.  Anything the engine does
// not send stays empty -- no value here is invented by the UI.
#pragma once

#include "shuyi/position.h"

#include <array>
#include <string>
#include <vector>

namespace shuyi {

struct AnalysisCandidateView {
    Move move{};
    double winrate = 0.5;
    int visits = 0;
    double averageScore = 0.0;
};

struct CellAnalysisView {
    Move move{};
    double winrate = 0.5;
    int visits = 0;
    double averageScore = 0.0;
    bool hasData = false;
    // The engine solved this move's subtree: win rate and score are exact, so a
    // small visit count does not turn them into a guess.
    bool proven = false;
    // This cell carries a real score: the exact margin of a solved line, or an
    // average that some visit measured.  Without it the cell shows "--"; 0.0
    // from "no score source" is not "the game is even".
    bool hasScore = false;
};

struct EngineAnalysisView {
    bool valid = false;
    bool hasScore = false;
    bool proven = false;
    std::string backend;
    std::string kind;
    std::string turn;
    std::string nn;
    Move best{};
    int depth = 0;
    long long nodes = 0;
    int visits = 0;
    double winrate = 0.5;
    int score = 0;
    double averageScore = 0.0;
    long long elapsedMs = 0;
    std::string pv;
    std::vector<AnalysisCandidateView> candidates;
    std::array<CellAnalysisView, kCells> cells{};
    std::array<bool, kCells> hasCell{};
};

double displayWinRate(const CellAnalysisView& cell);
std::vector<int> rankAnalysisCells(const EngineAnalysisView& view, const Position& position);
// The same ranking with symmetric equivalents collapsed to their best
// representative: for a list of moves, not for the board.
std::vector<int> distinctSymmetricCells(const std::vector<int>& ranked,
                                        const EngineAnalysisView& view,
                                        const Position& position);

// "c8:3" (the engine's compact form), "c8 3" (the protocol's move reply) or
// "pass".  Returns an invalid Move when the text is not a move.
Move parseMoveToken(const std::string& text);

// Parses the JSON object of an `info` line (without the `info ` prefix).
bool parseAnalysis(const std::string& json, EngineAnalysisView& out, std::string* error);

// Whether a fresh snapshot may replace the one cached for the position.  The
// window keeps what the engine said about a position after the user moves on, so
// a snapshot that searched less must not overwrite it: re-opening a record that
// carries numbers, or restarting the engine, begins again from a few visits.
bool analysisOutranks(const EngineAnalysisView& fresh, const EngineAnalysisView& cached);

} // namespace shuyi
