#include "shuyi/engine.h"
#include "shuyi/nn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace shuyi {
namespace {

using Clock = std::chrono::steady_clock;

struct Edge {
    Move move{};
    std::int32_t child = -1;
    // Index of the edge that owns the search for this move's symmetry orbit.  It
    // is the edge itself for a representative; a duplicate points at its
    // representative and is never visited, because the position after it is the
    // same game as the representative's, up to a rotation/reflection.
    std::int32_t representative = -1;
    std::uint32_t visits = 0;
    double valueSum = 0.0;
    double scoreSum = 0.0;
    // How many of those visits carried a real score (exact terminal result or a
    // trained score head).  Without one there is no score to report at all --
    // guessing it from the win probability produced numbers like -15 points on
    // an empty board.
    std::uint32_t scoreVisits = 0;
    float prior = 0.0f;
    bool searchable = true;
};

struct PathStep {
    int node = 0;
    int edge = 0;
    int child = 0;
};

class MctsEngine final : public Engine {
public:
    MctsEngine(const EngineOptions& options, bool useNn) : options_(options), useNn_(useNn), rng_(options.seed) {
        if (useNn_ && !options_.nnModelPath.empty()) {
            std::string error;
            nn_.load(options_.nnModelPath, &error);
        }
        useFasterNn_ = useNn_ && options_.nnOverride && options_.nnOverride->evaluate;
        resetTree();
    }

    EngineKind kind() const override { return useNn_ ? EngineKind::MctsNn : EngineKind::Mcts; }

    std::string infoTag() const override {
        if (!useNn_) return {};
        return "nn=" + (nn_.loaded() ? nn_.modelName() : std::string("none"));
    }

    std::vector<PolicyVisit> rootPolicy() const override {
        std::vector<PolicyVisit> result;
        if (nodesEmpty() || edgeCount_.empty()) return result;
        // A solved position stops searching after roughly one visit per move, so
        // the raw visit counts are almost uniform.  Handing that to self-play or
        // to a temperature sampler would make it pick a solved loss on purpose, so
        // a solution reports one move: the proven best.
        if (provenValue_[root_] >= 0.0) {
            int bestEdge = -1;
            for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
                const int e = firstEdge_[root_] + i;
                const Edge& edge = edges_[e];
                if (!edgeProven(edge)) continue;
                if (bestEdge < 0) {
                    bestEdge = e;
                    continue;
                }
                const Edge& best = edges_[bestEdge];
                if (edgeValue(edge) > edgeValue(best) + 1e-9 ||
                    (std::fabs(edgeValue(edge) - edgeValue(best)) <= 1e-9 &&
                     edgeScore(edge) > edgeScore(best) + 1e-9)) {
                    bestEdge = e;
                }
            }
            if (bestEdge >= 0) {
                result.push_back({edges_[bestEdge].move, static_cast<int>(visits_[root_]),
                                  edges_[bestEdge].prior});
                return result;
            }
        }
        for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
            const Edge& edge = edges_[firstEdge_[root_] + i];
            // Equivalent moves share one search, so they must share its count:
            // otherwise the policy target would teach the network that one of two
            // identical moves is better than the other.
            const Edge& owner = edges_[edge.representative];
            result.push_back({edge.move, static_cast<int>(owner.visits), edge.prior});
        }
        return result;
    }

    void reset() override { resetTree(); }

    void onPlay(const Position& position, Move move) override {
        if (nodesEmpty() || parent_.empty()) {
            resetTree();
            return;
        }
        if (edgeCount_[root_] == 0) return;
        for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
            const int e = firstEdge_[root_] + i;
            if (edges_[e].move == move && edges_[e].child >= 0) {
                root_ = edges_[e].child;
                rootKey_ = position.key();
                rootNoiseApplied_ = false;
                return;
            }
        }
        resetTree();
    }

    Analysis search(const Position& position, const SearchRequest& request) override {
        ensureRoot(position);
        if (position.gameOver()) return snapshot();
        if (options_.rootDirichletAlpha > 0.0 && !rootNoiseApplied_ && edgeCount_.size() > 0 &&
            edgeCount_[root_] > 1) {
            applyRootNoiseToEdges();
        }

        const auto started = Clock::now();
        auto lastReport = started - std::chrono::milliseconds(1000);
        const int timeLimit = std::max(0, request.timeLimitMs);
        const int visitLimit = std::max(0, request.visits);
        int simulated = 0;
        do {
            if (request.stop && request.stop->load(std::memory_order_relaxed)) break;
            simulate(position);
            ++simulated;
            // The position is solved with its exact margin: further visits cannot
            // change the result, so the rest of the budget stays unspent on
            // purpose.  A solved value with an unfinished margin keeps searching,
            // because the score is part of the answer.
            if (provenValue_[root_] >= 0.0 && scoreExact_[root_]) break;
            if (visitLimit > 0 && simulated >= visitLimit) break;
            const auto now = Clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastReport).count() >= 100) {
                lastReport = now;
                if (request.onUpdate) request.onUpdate(snapshot());
            }
            if (timeLimit > 0 &&
                std::chrono::duration_cast<std::chrono::milliseconds>(now - started).count() >= timeLimit) {
                break;
            }
        } while (true);

        Analysis analysis = snapshot();
        if (request.onUpdate) request.onUpdate(analysis);
        return analysis;
    }

private:
    bool nodesEmpty() const { return nodeMove_.empty(); }

    int newNode(int parent, Move move, bool terminal, std::uint8_t symmetry = 1) {
        const int index = static_cast<int>(nodeMove_.size());
        nodeMove_.push_back(move);
        parent_.push_back(parent);
        firstEdge_.push_back(0);
        edgeCount_.push_back(0);
        visits_.push_back(0);
        terminal_.push_back(terminal ? 1 : 0);
        deadEnd_.push_back(0);
        complete_.push_back(0);
        provenValue_.push_back(-1.0);
        provenScore_.push_back(0.0);
        scoreExact_.push_back(0);
        symmetry_.push_back(symmetry ? symmetry : 1);
        return index;
    }

    void resetTree() {
        nodeMove_.clear();
        parent_.clear();
        firstEdge_.clear();
        edgeCount_.clear();
        visits_.clear();
        terminal_.clear();
        deadEnd_.clear();
        complete_.clear();
        provenValue_.clear();
        provenScore_.clear();
        scoreExact_.clear();
        symmetry_.clear();
        edges_.clear();
        root_ = newNode(-1, Move{}, false);
        rootKey_ = 0;
        rootValid_ = false;
        rootNoiseApplied_ = false;
    }

    void ensureRoot(const Position& position) {
        // rootKey_ = 0 is also the *empty board's* key, so an explicit validity
        // flag is needed: without it the root of a fresh tree looked like it
        // already matched an empty board and kept the identity symmetry mask.
        if (!rootValid_ || rootKey_ != position.key()) {
            resetTree();
            rootKey_ = position.key();
            rootValid_ = true;
            // The tree is only worth exploring once per symmetry class: on an
            // empty or symmetric board half of the root moves are exact copies of
            // each other.
            symmetry_[root_] = position.symmetryMask();
        }
    }

    struct SimulationResult {
        double win = 0.5;
        double score = 0.0;
        bool hasScore = false;  // score is exact (terminal) or from a score head
    };

    SimulationResult outcomeFor(const Position& position, Player player) const {
        const double total = position.finalScore(options_.aValue);
        SimulationResult result;
        if (total > 0.0) result.win = player == Player::Red ? 1.0 : 0.0;
        else if (total < 0.0) result.win = player == Player::Black ? 1.0 : 0.0;
        else result.win = 0.5;
        result.score = player == Player::Red ? total : -total;
        result.hasScore = true;  // a finished game has an exact score
        return result;
    }

    SimulationResult rollout(Position position) const {
        const Player player = position.sideToMove();
        while (!position.gameOver()) {
            const auto moves = position.searchMoves();
            if (moves.empty()) break;
            std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
            position.play(moves[pick(rng_)]);
        }
        return outcomeFor(position, player);
    }

    // The network gives a win probability always, and a score only when the
    // model has a score head.  There is deliberately no "score from win rate"
    // fallback: mapping [0,1] onto +-73 points invented numbers (an empty board
    // came out at -15 points).  No score head -> no score, and the UI shows "--".
    SimulationResult blendNetValue(double value, double netScore, bool netHasScore,
                                   const Position& position) const {
        const double blend = std::clamp(options_.nnValueBlend, 0.0, 1.0);
        SimulationResult result;
        if (blend >= 1.0) {
            result.win = value;
            result.score = netScore;
            result.hasScore = netHasScore;
            return result;
        }
        const SimulationResult playout = rollout(position);
        result.win = blend * value + (1.0 - blend) * playout.win;
        if (netHasScore && playout.hasScore) {
            result.score = netScore * blend + playout.score * (1.0 - blend);
            result.hasScore = true;
        } else if (netHasScore) {
            result.score = netScore;
            result.hasScore = true;
        } else if (playout.hasScore) {
            result.score = playout.score;
            result.hasScore = true;
        }
        return result;
    }

    std::vector<Move> selectExpansionMoves(int node, const Position& position,
                                           std::uint8_t symmetry,
                                           std::vector<float>& priors,
                                           std::vector<std::uint8_t>& searchable,
                                           std::vector<int>& representatives,
                                           double* valueOut, bool* hasValue,
                                           double* scoreOut = nullptr,
                                           bool* hasScore = nullptr,
                                           bool* complete = nullptr) const {
        std::vector<Move> moves = position.searchMoves();
        priors.assign(moves.size(), 1.0f / static_cast<float>(std::max<std::size_t>(1, moves.size())));
        searchable.assign(moves.size(), 1);
        representatives.clear();
        if (complete) *complete = false;
        if (moves.empty()) return moves;

        // A forced pass has no policy entry; keep its prior at 1 (it is the only
        // move) but still take the value from the network.
        const bool forcedPass = moves.size() == 1 && isPass(moves[0]);

        // A network trained for the other rule set would evaluate with the
        // wrong value semantics, so it is ignored rather than trusted.
        if (useNn_ && (nn_.loaded() || useFasterNn_) && !forcedPass &&
            nn_.ruleset() == position.ruleset()) {
            const NnOutput output = (useFasterNn_ && options_.nnOverride->evaluate)
                                        ? options_.nnOverride->evaluate(position)
                                        : nn_.evaluate(position);
            if (valueOut) *valueOut = output.value;
            if (hasValue) *hasValue = true;
            // The score head is an extra target; when the model has one its
            // prediction replaces the value-to-score heuristic downstream.
            if (hasScore) {
                *hasScore = (useFasterNn_ && options_.nnOverride->hasScoreHead)
                                ? options_.nnOverride->hasScoreHead()
                                : nn_.hasScoreHead();
            }
            if (scoreOut) *scoreOut = output.expectedScore;
            float maxLogit = -std::numeric_limits<float>::infinity();
            std::vector<float> weights(moves.size());
            for (std::size_t i = 0; i < moves.size(); ++i) {
                weights[i] = output.policy[nn_.moveIndex(moves[i])];
                maxLogit = std::max(maxLogit, weights[i]);
            }
            double sum = 0.0;
            for (float& w : weights) {
                w = std::exp(w - maxLogit);
                sum += w;
            }
            if (!(sum > 0.0) || !std::isfinite(sum)) {
                for (float& w : weights) w = 1.0f;
                sum = static_cast<double>(weights.size());
            }
            const double blend = std::clamp(options_.nnPriorBlend, 0.0, 1.0);
            const float uniform = 1.0f / static_cast<float>(moves.size());
            for (std::size_t i = 0; i < weights.size(); ++i) {
                priors[i] = static_cast<float>(
                    blend * (weights[i] / sum) + (1.0 - blend) * uniform);
            }
        }

        // Group the moves into symmetry orbits.  `orbit[i]` is the index of the
        // move that owns the search for i's class (i itself when it is the
        // representative).  Two moves related by a rotation/reflection of the
        // current position lead to the same game, so the duplicate is never
        // searched -- it only mirrors the representative's numbers in the report.
        std::vector<int> orbit(moves.size());
        for (std::size_t i = 0; i < moves.size(); ++i) orbit[i] = static_cast<int>(i);
        if (symmetry > 1) {
            std::array<int, kCells * 10> indexOf{};
            indexOf.fill(-1);
            for (std::size_t i = 0; i < moves.size(); ++i) {
                indexOf[moves[i].cell * 10 + moves[i].number] = static_cast<int>(i);
            }
            for (std::size_t i = 0; i < moves.size(); ++i) {
                if (orbit[i] != static_cast<int>(i)) continue;
                for (int s = 0; s < Position::kSymmetryCount; ++s) {
                    if (((symmetry >> s) & 1u) == 0) continue;
                    const int image = indexOf[Position::transformCell(moves[i].cell, s) * 10 +
                                              moves[i].number];
                    if (image > static_cast<int>(i)) orbit[image] = static_cast<int>(i);
                }
            }
            // The member that leads the class is the one the policy likes best;
            // the network's policy is only approximately symmetric, so taking the
            // lowest index could throw away the best-rated representative.
            std::vector<int> leader(moves.size());
            for (std::size_t i = 0; i < moves.size(); ++i) leader[i] = static_cast<int>(i);
            for (std::size_t i = 0; i < moves.size(); ++i) {
                const int owner = orbit[i];
                const int best = leader[owner];
                if (i != static_cast<std::size_t>(owner) && priors[i] > priors[best]) {
                    leader[owner] = static_cast<int>(i);
                }
            }
            for (std::size_t i = 0; i < moves.size(); ++i) {
                orbit[i] = leader[orbit[i]];
            }
            for (std::size_t i = 0; i < moves.size(); ++i) {
                if (orbit[i] != static_cast<int>(i)) searchable[i] = 0;
            }
        }

        // Endgame: do not prune.  A finished game has an exact result, and that
        // result replaces the network estimate whenever the search reaches it --
        // but only if the tree is allowed to expand into the last plies.
        const int pliesLeft = kTotalMoves - position.ply();
        if (options_.mctsFullWidthPlies > 0 && pliesLeft <= options_.mctsFullWidthPlies) {
            if (complete) *complete = true;
            representatives = orbit;
            return moves;
        }

        const int maxChildren = std::max(1, node == root_ ? options_.mctsRootChildren
                                                          : options_.mctsMaxChildren);
        if (static_cast<int>(moves.size()) <= maxChildren) {
            if (complete) *complete = true;
            representatives = orbit;
            return moves;
        }

        if (node == root_) {
            // Keep all root moves for display and policy targets, but only a
            // bounded subset is selectable during search.  This avoids the
            // old 729-root-move bottleneck while still exposing every cell.
            std::vector<int> order(moves.size());
            std::iota(order.begin(), order.end(), 0);
            if (useNn_ && nn_.loaded()) {
                std::stable_sort(order.begin(), order.end(),
                                 [&](int a, int b) { return priors[a] > priors[b]; });
            } else {
                std::shuffle(order.begin(), order.end(), rng_);
            }
            std::fill(searchable.begin(), searchable.end(), 0);
            int chosen = 0;
            for (std::size_t i = 0; i < order.size() && chosen < maxChildren; ++i) {
                const int candidate = order[i];
                if (orbit[candidate] != candidate) continue;  // represented elsewhere
                searchable[candidate] = 1;
                ++chosen;
            }
            representatives = orbit;
            return moves;
        }

        if (useNn_ && nn_.loaded()) {
            std::vector<int> order(moves.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return priors[a] > priors[b]; });
            std::vector<Move> selected;
            std::vector<float> selectedPriors;
            selected.reserve(maxChildren);
            selectedPriors.reserve(maxChildren);
            for (std::size_t i = 0; i < order.size() && static_cast<int>(selected.size()) < maxChildren; ++i) {
                const int candidate = order[i];
                if (orbit[candidate] != candidate) continue;
                selected.push_back(moves[candidate]);
                selectedPriors.push_back(priors[candidate]);
            }
            priors.swap(selectedPriors);
            searchable.assign(selected.size(), 1);
            representatives.resize(selected.size());
            for (std::size_t i = 0; i < selected.size(); ++i) representatives[i] = static_cast<int>(i);
            return selected;
        }

        std::vector<std::size_t> keep;
        keep.reserve(moves.size());
        for (std::size_t i = 0; i < moves.size(); ++i) {
            if (orbit[i] == static_cast<int>(i)) keep.push_back(i);
        }
        std::shuffle(keep.begin(), keep.end(), rng_);
        if (static_cast<int>(keep.size()) > maxChildren) keep.resize(maxChildren);
        std::vector<Move> selected;
        selected.reserve(keep.size());
        for (std::size_t index : keep) selected.push_back(moves[index]);
        moves.swap(selected);
        priors.assign(moves.size(), 1.0f / static_cast<float>(std::max<std::size_t>(1, moves.size())));
        searchable.assign(moves.size(), 1);
        representatives.resize(moves.size());
        for (std::size_t i = 0; i < moves.size(); ++i) representatives[i] = static_cast<int>(i);
        return moves;
    }

    void applyRootNoiseToPriors(std::vector<float>& priors) {
        if (priors.size() < 2) return;
        std::gamma_distribution<double> gamma(options_.rootDirichletAlpha, 1.0);
        std::vector<double> noise(priors.size());
        double sum = 0.0;
        for (double& value : noise) {
            value = gamma(rng_);
            sum += value;
        }
        const double weight = std::clamp(options_.rootNoiseWeight, 0.0, 1.0);
        if (sum > 0.0) {
            for (std::size_t i = 0; i < priors.size(); ++i) {
                priors[i] = static_cast<float>((1.0 - weight) * priors[i] +
                                               weight * (noise[i] / sum));
            }
        }
    }

    void applyRootNoiseToEdges() {
        if (edgeCount_[root_] < 2) return;
        std::vector<float> priors(edgeCount_[root_]);
        for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
            priors[i] = edges_[firstEdge_[root_] + i].prior;
        }
        applyRootNoiseToPriors(priors);
        for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
            edges_[firstEdge_[root_] + i].prior = priors[i];
        }
        rootNoiseApplied_ = true;
    }

    bool expand(int node, const Position& position, double* valueOut, double* scoreOut = nullptr,
                bool* hasScore = nullptr) {
        if (edgeCount_[node] != 0 || terminal_[node] || deadEnd_[node]) return false;
        if (edges_.size() >= options_.mctsMaxTreeEdges) {
            // Keep the tree bounded during very long analysis sessions.  The
            // node then behaves like a rollout leaf instead of allocating new
            // children forever -- but it is NOT terminal, so its static score
            // must never be reported as a hard result.
            deadEnd_[node] = 1;
            return false;
        }
        std::vector<float> priors;
        std::vector<std::uint8_t> searchable;
        std::vector<int> representatives;
        double value = 0.0;
        bool hasValue = false;
        bool complete = false;
        std::vector<Move> moves = selectExpansionMoves(node, position, symmetry_[node], priors,
                                                       searchable, representatives, &value, &hasValue,
                                                       scoreOut, hasScore, &complete);
        if (moves.empty()) {
            deadEnd_[node] = 1;
            return false;
        }
        complete_[node] = complete ? 1 : 0;
        if (node == root_ && !rootNoiseApplied_ && options_.rootDirichletAlpha > 0.0) {
            applyRootNoiseToPriors(priors);
            rootNoiseApplied_ = true;
        }
        firstEdge_[node] = static_cast<int>(edges_.size());
        edgeCount_[node] = static_cast<std::uint32_t>(moves.size());
        for (std::size_t i = 0; i < moves.size(); ++i) {
            Edge edge;
            edge.move = moves[i];
            edge.prior = priors[i];
            edge.searchable = searchable[i] != 0;
            edge.representative =
                firstEdge_[node] + (i < representatives.size() ? representatives[i]
                                                               : static_cast<int>(i));
            edges_.push_back(edge);
        }
        if (valueOut && hasValue) *valueOut = value;
        return hasValue;
    }

    int chooseUntried(int node) {
        int best = -1;
        float bestPrior = -1.0f;
        const int begin = firstEdge_[node];
        const int end = begin + static_cast<int>(edgeCount_[node]);
        for (int e = begin; e < end; ++e) {
            if (!edges_[e].searchable || edges_[e].child >= 0) continue;
            if (best < 0 || edges_[e].prior > bestPrior) {
                best = e;
                bestPrior = edges_[e].prior;
            }
        }
        return best;
    }

    int chooseChild(int node) {
        int best = -1;
        double bestScore = -std::numeric_limits<double>::infinity();
        const int begin = firstEdge_[node];
        const int end = begin + static_cast<int>(edgeCount_[node]);
        const double logVisits = std::log(static_cast<double>(visits_[node]) + 1.0);
        for (int e = begin; e < end; ++e) {
            Edge& edge = edges_[e];
            if (!edge.searchable || edge.child < 0 || edge.visits == 0) continue;
            const double mean = edge.valueSum / edge.visits;
            double exploration;
            if (useNn_) {
                exploration = options_.mctsPuct * edge.prior * std::sqrt(static_cast<double>(visits_[node]) + 1.0) /
                              (1.0 + edge.visits);
            } else {
                exploration = options_.mctsC * std::sqrt(logVisits / edge.visits);
            }
            const double score = mean + exploration;
            if (score > bestScore) {
                bestScore = score;
                best = e;
            }
        }
        return best;
    }

    void simulate(const Position& rootPosition) {
        Position position = rootPosition;
        std::vector<int> nodePath{root_};
        std::vector<PathStep> path;
        int node = root_;

        while (true) {
            expand(node, position, nullptr);
            const int untried = chooseUntried(node);
            if (untried >= 0) {
                Edge& edge = edges_[untried];
                const std::uint8_t childSymmetry =
                    static_cast<std::uint8_t>(symmetry_[node] &
                                              Position::stabilizer(edge.move.cell, symmetry_[node]));
                position.play(edge.move);
                const int child = newNode(node, edge.move, position.gameOver(), childSymmetry);
                edge.child = child;
                path.push_back({node, untried, child});
                node = child;
                nodePath.push_back(child);
                break;
            }
            const int chosen = chooseChild(node);
            if (chosen < 0) break;
            Edge& edge = edges_[chosen];
            position.play(edge.move);
            path.push_back({node, chosen, edge.child});
            node = edge.child;
            nodePath.push_back(node);
            if (terminal_[node]) break;
        }

        // One network call per simulation: expanding the leaf already computes
        // its value, so there is no second evaluation of the same position.
        SimulationResult result;
        double leafValue = 0.0;
        double leafScore = 0.0;
        bool leafHasScore = false;
        const bool evaluated =
            !terminal_[node] && expand(node, position, &leafValue, &leafScore, &leafHasScore);
        if (terminal_[node]) {
            result = outcomeFor(position, position.sideToMove());
            // A finished game is proven by the rules: no search can change it.
            provenValue_[node] = result.win;
            provenScore_[node] = result.score;
            scoreExact_[node] = 1;
        } else if (evaluated) {
            result = blendNetValue(leafValue, leafScore, leafHasScore, position);
        } else {
            result = rollout(position);
        }
        double value = result.win;
        double score = result.score;
        bool hasScore = result.hasScore;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            value = 1.0 - value;
            score = -score;
            Edge& edge = edges_[it->edge];
            edge.visits++;
            edge.valueSum += value;
            if (hasScore) {
                edge.scoreSum += score;
                ++edge.scoreVisits;
            }
        }
        for (int n : nodePath) ++visits_[n];
        // Now let the proof climb: a terminal leaf can make its parent a forced
        // win, that parent can make the grandparent one, and so on.  Stop at the
        // first node that stays unsolved -- its parent cannot be solved either.
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            if (!updateProof(it->node)) break;
            if (provenValue_[root_] >= 0.0) break;
        }
    }

    // Try to solve `node` from its children.  Returns true when the node is (or
    // already was) proven, i.e. when its parent may now be provable too.  A node
    // that is already proven is re-examined so the margin can become exact later.
    // Only orbit representatives are examined: a duplicate leads to the same game
    // as its representative, so it can neither add nor remove a proof.
    bool updateProof(int node) {
        if (terminal_[node]) return true;
        if (deadEnd_[node] || edgeCount_[node] == 0) return false;
        if (provenValue_[node] < 0.0) {
            // A single reply that is a proven loss for the opponent is a forced
            // win: no other reply matters, and the move list need not be complete.
            int winner = -1;
            double winnerScore = 0.0;
            for (int i = 0; i < static_cast<int>(edgeCount_[node]); ++i) {
                const int e = firstEdge_[node] + i;
                if (edges_[e].representative != e) continue;
                const Edge& edge = edges_[e];
                if (edge.child < 0) continue;
                const double child = provenValue_[edge.child];
                if (child < 0.0 || child > 0.0 + 1e-9) continue;
                // The child lost, so this move wins.  Among several wins keep the
                // one with the best (highest) score for us.
                const double candidate = -provenScore_[edge.child];
                if (winner < 0 || candidate > winnerScore) {
                    winner = edge.child;
                    winnerScore = candidate;
                }
            }
            if (winner >= 0) {
                provenValue_[node] = 1.0;
                provenScore_[node] = winnerScore;
                // The margin is only final when no remaining reply can beat it, so
                // it stays private until recomputeProvenScore has seen every reply.
                // Publishing it here produced "+3" on a position whose true best
                // reply was worth "+6".
                scoreExact_[node] = 0;
            } else {
                // Otherwise every reply must exist and be proven before "all
                // replies are bad" can be concluded.  That needs the complete move
                // list: a node whose width was pruned has replies the tree never
                // generated.
                if (!complete_[node]) return false;
                for (int i = 0; i < static_cast<int>(edgeCount_[node]); ++i) {
                    const int e = firstEdge_[node] + i;
                    if (edges_[e].representative != e) continue;  // covered by its class
                    const Edge& edge = edges_[e];
                    if (edge.child < 0 || provenValue_[edge.child] < 0.0) return false;
                }
                recomputeProvenScore(node);
                return true;
            }
        }
        // Solved in value; the margin may still become exact once every reply is.
        if (!scoreExact_[node] && complete_[node]) recomputeProvenScore(node);
        return true;
    }

    // Maximin over every reply; does nothing unless all replies are proven.
    void recomputeProvenScore(int node) {
        double bestValue = -1.0;
        double bestScore = 0.0;
        for (int i = 0; i < static_cast<int>(edgeCount_[node]); ++i) {
            const int e = firstEdge_[node] + i;
            if (edges_[e].representative != e) continue;
            const Edge& edge = edges_[e];
            if (edge.child < 0 || provenValue_[edge.child] < 0.0) return;
            const double value = 1.0 - provenValue_[edge.child];
            const double score = -provenScore_[edge.child];
            if (value > bestValue + 1e-9 ||
                (std::fabs(value - bestValue) <= 1e-9 && score > bestScore)) {
                bestValue = value;
                bestScore = score;
            }
        }
        provenValue_[node] = bestValue;
        provenScore_[node] = bestScore;
        // The margin is exact when every reply that reaches this result has an
        // exact margin of its own.
        bool exact = true;
        for (int i = 0; i < static_cast<int>(edgeCount_[node]); ++i) {
            const int e = firstEdge_[node] + i;
            if (edges_[e].representative != e) continue;
            const Edge& edge = edges_[e];
            const double value = 1.0 - provenValue_[edge.child];
            if (std::fabs(value - bestValue) <= 1e-9 && !scoreExact_[edge.child]) {
                exact = false;
                break;
            }
        }
        scoreExact_[node] = exact ? 1 : 0;
    }

    std::string pvFromRoot() const {
        if (nodesEmpty() || edgeCount_[root_] == 0) return {};
        std::string pv;
        int node = root_;
        for (int depth = 0; depth < 8; ++depth) {
            int best = -1;
            std::uint32_t bestVisits = 0;
            double bestValue = -1.0;
            for (int i = 0; i < static_cast<int>(edgeCount_[node]); ++i) {
                const int e = firstEdge_[node] + i;
                const Edge& edge = edges_[e];
                if (edge.child < 0) continue;
                // Inside a solved tree the ordered visits no longer mean anything:
                // follow the proven best line instead.
                if (provenValue_[node] >= 0.0) {
                    const double value = edgeProven(edge) ? edgeValue(edge) : -1.0;
                    if (best < 0 || value > bestValue + 1e-9) {
                        best = e;
                        bestValue = value;
                    }
                    continue;
                }
                if (edge.visits > bestVisits) {
                    bestVisits = edge.visits;
                    best = e;
                }
            }
            if (best < 0) break;
            if (!pv.empty()) pv += ",";
            pv += squareName(edges_[best].move.cell) + ":" + std::to_string(edges_[best].move.number);
            node = edges_[best].child;
            if (terminal_[node]) break;
        }
        return pv;
    }

    // Value of an edge for the side to move at its parent.  A solved subtree
    // returns the exact result; anything else returns the sampled average, so an
    // unproven line can never be reported as a certainty.
    bool edgeProven(const Edge& edge) const {
        return edge.child >= 0 && provenValue_[edge.child] >= 0.0;
    }
    bool edgeScoreExact(const Edge& edge) const {
        return edge.child >= 0 && scoreExact_[edge.child] != 0;
    }
    double edgeValue(const Edge& edge) const {
        if (edgeProven(edge)) return 1.0 - provenValue_[edge.child];
        return edge.visits ? edge.valueSum / edge.visits : 0.5;
    }
    bool edgeHasScore(const Edge& edge) const {
        if (edgeProven(edge)) return edgeScoreExact(edge);
        return edge.scoreVisits > 0;
    }
    double edgeScore(const Edge& edge) const {
        // A proven line with an unfinished margin reports nothing rather than a
        // lower bound that looks like a result.
        if (edgeProven(edge)) return edgeScoreExact(edge) ? -provenScore_[edge.child] : 0.0;
        return edge.scoreVisits ? edge.scoreSum / static_cast<double>(edge.scoreVisits) : 0.0;
    }

    Analysis snapshot() const {
        Analysis analysis;
        analysis.kind = kind();
        if (nodesEmpty() || edgeCount_[root_] == 0) return analysis;

        int bestEdge = -1;
        for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
            const int e = firstEdge_[root_] + i;
            const Edge& edge = edges_[e];
            if (edge.child < 0) continue;
            if (bestEdge < 0) {
                bestEdge = e;
                continue;
            }
            const Edge& best = edges_[bestEdge];
            // A proven result outranks any sampled average: never play a ~99%
            // guess over a forced win.  Among solutions prefer the best score,
            // among estimates the most visited move.
            if (edgeProven(edge) != edgeProven(best)) {
                if (edgeProven(edge)) bestEdge = e;
            } else if (edgeProven(edge)) {
                if (edgeValue(edge) > edgeValue(best) + 1e-9 ||
                    (std::fabs(edgeValue(edge) - edgeValue(best)) <= 1e-9 &&
                     edgeScore(edge) > edgeScore(best) + 1e-9)) {
                    bestEdge = e;
                }
            } else if (edge.visits > best.visits) {
                bestEdge = e;
            }
        }
        if (bestEdge < 0) return analysis;

        const bool rootProven = provenValue_[root_] >= 0.0;
        analysis.valid = true;
        analysis.proven = rootProven;
        analysis.best = edges_[bestEdge].move;
        // Simulations actually run, not the sum of root-edge visits: the first
        // simulation only expands the root and lands on no edge, so the edge sum
        // is one short.  A rating needs "how much search did this move get",
        // which is exactly the number of simulations.
        analysis.visits = static_cast<int>(visits_[root_]);
        analysis.nodes = static_cast<long long>(edges_.size());
        analysis.depth = 1;
        // The position's win rate is the value of the line the engine would play,
        // not an average over moves it rejects.
        analysis.winrate = rootProven ? provenValue_[root_] : edgeValue(edges_[bestEdge]);
        if (rootProven) {
            // The margin is published only when it is a true minimax margin: the
            // whole tree when that is solved, otherwise the margin of the line the
            // engine would actually play.
            analysis.hasScore = scoreExact_[root_] != 0 || edgeScoreExact(edges_[bestEdge]);
            if (!analysis.hasScore) {
                analysis.averageScore = 0.0;
            } else if (scoreExact_[root_]) {
                analysis.averageScore = provenScore_[root_];
            } else {
                analysis.averageScore = edgeScore(edges_[bestEdge]);
            }
        } else {
            // A score is reported only when some visits carried a real one: an
            // exact terminal result or a trained score head.  Otherwise the third
            // row of the UI stays empty instead of showing a guess.
            double weighted = 0.0;
            double weight = 0.0;
            for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
                const Edge& edge = edges_[firstEdge_[root_] + i];
                if (!edgeHasScore(edge)) continue;
                const double w = std::max(1.0, static_cast<double>(edge.visits));
                weighted += edgeScore(edge) * w;
                weight += w;
            }
            analysis.hasScore = weight > 0.0;
            analysis.averageScore = weight > 0.0 ? weighted / weight : 0.0;
        }
        analysis.score = analysis.hasScore
                             ? static_cast<int>(std::lround(analysis.averageScore))
                             : 0;
        analysis.pv = pvFromRoot();

        std::vector<int> order;
        for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
            const int e = firstEdge_[root_] + i;
            if (edges_[e].child >= 0) order.push_back(e);
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const bool provenA = edgeProven(edges_[a]);
            const bool provenB = edgeProven(edges_[b]);
            if (provenA != provenB) return provenA;
            if (provenA) return edgeValue(edges_[a]) > edgeValue(edges_[b]);
            if (edges_[a].visits != edges_[b].visits) return edges_[a].visits > edges_[b].visits;
            return edges_[a].valueSum > edges_[b].valueSum;
        });
        const int top = std::min<int>(8, static_cast<int>(order.size()));
        for (int i = 0; i < top; ++i) {
            const Edge& edge = edges_[order[i]];
            AnalysisCandidate candidate;
            candidate.move = edge.move;
            candidate.visits = static_cast<int>(edge.visits);
            candidate.winrate = edgeValue(edge);
            candidate.averageScore = edgeScore(edge);
            candidate.score = static_cast<int>(std::lround(candidate.averageScore));
            analysis.candidates.push_back(candidate);
        }

        // One recommendation for every empty cell, even if that particular
        // (cell, number) edge was not selected for search.  Unvisited cells
        // retain their prior and are marked hasData=false.  A cell whose move is a
        // symmetry duplicate of a searched one reports the representative's
        // numbers, so equivalent points show an identical result instead of zero
        // visits and a blank mark.
        for (int cell = 0; cell < kCells; ++cell) {
            int chosen = -1;
            for (int i = 0; i < static_cast<int>(edgeCount_[root_]); ++i) {
                const int e = firstEdge_[root_] + i;
                if (edges_[e].move.cell != cell) continue;
                const Edge& owner = edges_[edges_[e].representative];
                const Edge& incumbent = chosen >= 0 ? edges_[edges_[chosen].representative] : edges_[e];
                if (chosen < 0 || owner.visits > incumbent.visits ||
                    (owner.visits == incumbent.visits && edges_[e].prior > edges_[chosen].prior)) {
                    chosen = e;
                }
            }
            if (chosen < 0) continue;
            const Edge& edge = edges_[edges_[chosen].representative];
            CellAnalysis cellInfo;
            cellInfo.cell = cell;
            // By symmetry the equivalent move carries the same number at this
            // cell; the transforms preserve stone numbers.
            cellInfo.move = Move(cell, edge.move.number);
            cellInfo.visits = static_cast<int>(edge.visits);
            cellInfo.winrate = edgeValue(edge);
            cellInfo.averageScore = edgeScore(edge);
            cellInfo.hasData = edge.visits > 0;
            cellInfo.proven = edgeProven(edge);
            analysis.cells.push_back(cellInfo);
        }
        return analysis;
    }

    EngineOptions options_;
    bool useNn_ = false;
    bool useFasterNn_ = false;  // a GPU (or other) evaluator replaces the CPU forward
    mutable std::mt19937_64 rng_;
    PolicyValueNet nn_;

    // SoA node arrays.
    std::vector<Move> nodeMove_;
    std::vector<std::int32_t> parent_;
    std::vector<std::int32_t> firstEdge_;
    std::vector<std::uint32_t> edgeCount_;
    std::vector<std::uint32_t> visits_;
    // The game is over here: the exact result is known from the rules alone.
    std::vector<std::uint8_t> terminal_;
    // Expansion was refused (tree size cap).  The position is NOT over, so the
    // static score must never be presented as a terminal result.
    std::vector<std::uint8_t> deadEnd_;
    // All legal moves were expanded (nothing was pruned by width).  Only then
    // can "every reply loses for the opponent" be concluded -- a proof is only
    // valid over a complete move list.
    std::vector<std::uint8_t> complete_;
    // Proven ("solved") nodes.  provenValue_ is the exact game-theoretic result
    // for the side to move and provenScore_ the exact terminal score that the
    // best defence leads to; both are undefined (-1) until the node is proven.
    // A terminal leaf is proven by the rules, and a node is proven as soon as
    // one reply is a proven loss for the opponent, or every reply is proven.
    // Until 2026-11 a leaf value was all there was, which is exactly why the
    // engine could claim ~99% on a line whose refutation it had never searched.
    std::vector<double> provenValue_;
    std::vector<double> provenScore_;
    // Whether provenScore_ is the true minimax margin rather than the best one
    // seen so far.  A win proven through one reply is exact in value but its
    // margin can still improve when the other replies are solved, so the score is
    // only published once every reply that achieves the result is solved too.
    std::vector<std::uint8_t> scoreExact_;
    // Bitmask of the symmetries that map this node's position onto itself.
    std::vector<std::uint8_t> symmetry_;
    std::vector<Edge> edges_;
    int root_ = 0;
    std::uint64_t rootKey_ = 0;
    bool rootValid_ = false;
    bool rootNoiseApplied_ = false;
};

} // namespace

std::unique_ptr<Engine> makeMctsEngine(const EngineOptions& options, bool useNn) {
    return std::make_unique<MctsEngine>(options, useNn);
}

std::unique_ptr<Engine> makeMctsNnEngine(const EngineOptions& options) {
    EngineOptions adjusted = options;
    // NN inference is much more expensive than a random rollout.  A smaller
    // root width keeps short time controls from spending every visit on the
    // root's children.  Self-play can raise this when it has a larger budget.
    adjusted.mctsRootChildren =
        std::min(adjusted.mctsRootChildren, std::max(1, adjusted.mctsNnRootChildren));
    return std::make_unique<MctsEngine>(adjusted, true);
}

} // namespace shuyi
