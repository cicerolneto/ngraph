//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <regex>

#include "matcher.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace
        {
            class MatchStateImp : public MatchState
            {
            public:
                MatchStateImp(Matcher& matcher)
                    : m_matcher(matcher)
                {
                }

                PatternValueMap& get_pattern_map() override { return m_pattern_value_map; }
                void match_value(const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override
                {
                    m_matcher.match_value(pattern_value, graph_value, m_pattern_value_map);
                }
                void match_inputs(const Output<Node>& pattern_value,
                                  const Output<Node>& graph_value) override
                {
                    m_matcher.match_arguments(pattern_value, graph_value, m_pattern_value_map);
                }
                void start_match() override { m_saved_maps.push(m_pattern_value_map); }
                void abort_match() override
                {
                    m_pattern_value_map = m_saved_maps.top();
                    m_saved_maps.pop();
                }
                void finish_match() override { m_saved_maps.pop(); }
            protected:
                Matcher& m_matcher;
                PatternValueMap m_pattern_value_map;
                std::stack<PatternValueMap> m_saved_maps;
            };
        }

        PatternMap Matcher::get_pattern_map() const { return as_pattern_map(m_pattern_map); }
        size_t Matcher::add_node(Output<Node> node)
        {
            size_t result = m_matched_list.size();
            m_matched_list.push_back(node.get_node_shared_ptr());
            return result;
        }

        bool Matcher::abort_match(size_t watermark, bool matched)
        {
            if (!matched)
            {
                m_matched_list.erase(m_matched_list.begin() + watermark, m_matched_list.end());
            }
            return matched;
        }

        std::shared_ptr<Node> Matcher::get_match_root()
        {
            return m_match_root.get_node_shared_ptr();
        }

        Output<Node> Matcher::get_match_value() { return m_match_root; }
        bool Matcher::is_contained_match(const NodeVector& exclusions, bool ignore_unused)
        {
            if (exclusions.empty())
            {
                NodeVector label_exclusions;
                for (auto entry : m_pattern_map)
                {
                    // leaf label
                    if (entry.first.get_node_shared_ptr()->get_input_size() == 0)
                    {
                        label_exclusions.push_back(entry.second.get_node_shared_ptr());
                    }
                }
                return ngraph::get_subgraph_outputs(
                           get_matched_nodes(), label_exclusions, ignore_unused)
                           .size() < 2;
            }

            return ngraph::get_subgraph_outputs(get_matched_nodes(), exclusions).size() < 2;
        }

        bool Matcher::match_value(const ngraph::Output<Node>& pattern_value,
                                  const ngraph::Output<Node>& graph_value,
                                  PatternValueMap& pattern_map)
        {
            ngraph::Output<Node> real_pattern_value =
                m_follow_goe &&
                        is_type<ngraph::op::GetOutputElement>(pattern_value.get_node_shared_ptr())
                    ? pattern_value.get_node_shared_ptr()->input_value(0)
                    : pattern_value;
            ngraph::Output<Node> real_graph_value =
                m_follow_goe &&
                        is_type<ngraph::op::GetOutputElement>(graph_value.get_node_shared_ptr())
                    ? graph_value.get_node_shared_ptr()->input_value(0)
                    : graph_value;

            if (real_pattern_value.get_index() != real_graph_value.get_index() ||
                (is_strict_mode() && (!real_pattern_value.get_element_type().compatible(
                                          real_graph_value.get_element_type()) ||
                                      !real_pattern_value.get_partial_shape().compatible(
                                          real_graph_value.get_partial_shape()))))
            {
                return false;
            }
            std::shared_ptr<Node> pattern_node = real_pattern_value.get_node_shared_ptr();
            std::shared_ptr<Node> graph_node = real_graph_value.get_node_shared_ptr();

            // This env var allows one to specify node name patterns to abort pattern matching
            // at particular nodes. The upshot is that one can quickly zero in on an offending
            // fusion by disabling individual fusions or optimizations that use Matcher.
            static const char* node_skip_cregex = std::getenv("NGRAPH_FAIL_MATCH_AT");
            if (node_skip_cregex)
            {
                static const std::regex node_skip_regex(node_skip_cregex);
                if (std::regex_match(graph_node->get_name(), node_skip_regex))
                {
                    NGRAPH_DEBUG << "[MATCHER] Aborting at " << *graph_node
                                 << " due to NGRAPH_MATCHER_SKIP set to " << node_skip_cregex;
                    return false;
                }
            }

            auto watermark = add_node(real_graph_value);

            if (pattern_node->is_pattern())
            {
                return abort_match(
                    watermark,
                    std::static_pointer_cast<op::Pattern>(pattern_node)
                        ->match_value(*this, real_pattern_value, real_graph_value, pattern_map));
            }

            if (pattern_node->get_type_info() == graph_node->get_type_info())
            {
                return abort_match(
                    watermark, match_arguments(real_pattern_value, real_graph_value, pattern_map));
            }

            NGRAPH_DEBUG << "[MATCHER] Aborting at " << *graph_node << " for pattern "
                         << *pattern_node;
            return abort_match(watermark, false);
        }

        bool Matcher::match_permutation(const OutputVector& pattern_args,
                                        const OutputVector& args,
                                        PatternValueMap& pattern_map)
        {
            m_depth++;
            for (size_t i = 0; i < args.size(); i++)
            {
                if (!match_value(pattern_args.at(i), args.at(i), pattern_map))
                {
                    m_depth--;
                    return false;
                }
            }
            m_depth--;
            return true;
        }

        bool Matcher::match_arguments(const Output<Node>& pattern_value,
                                      const Output<Node>& graph_value,
                                      PatternValueMap& pattern_map)
        {
            auto pattern_node = pattern_value.get_node_shared_ptr();
            auto graph_node = graph_value.get_node_shared_ptr();
            NGRAPH_DEBUG << "[MATCHER] Match arguments at " << *graph_node << " for pattern "
                         << *pattern_node;

            auto args = graph_node->input_values();
            auto pattern_args = pattern_node->input_values();

            if (args.size() != pattern_args.size())
            {
                NGRAPH_DEBUG << "[MATCHER] Aborting at " << *graph_node << " for pattern "
                             << *pattern_node;
                return false;
            }

            if (graph_node->is_commutative())
            {
                // TODO: [nikolayk] we don't really have to use lexicographically-based perms,
                // heap's algo should be faster
                std::sort(begin(pattern_args),
                          end(pattern_args),
                          [](const ngraph::Output<ngraph::Node>& n1,
                             const ngraph::Output<ngraph::Node>& n2) { return n1 < n2; });
                do
                {
                    PatternValueMap copy{pattern_map};
                    if (match_permutation(pattern_args, args, copy))
                    {
                        pattern_map.insert(begin(copy), end(copy));
                        return true;
                    }
                } while (std::next_permutation(
                    begin(pattern_args),
                    end(pattern_args),
                    [](const ngraph::Output<ngraph::Node>& n1,
                       const ngraph::Output<ngraph::Node>& n2) { return n1 < n2; }));
            }
            else
            {
                PatternValueMap copy{pattern_map};
                if (match_permutation(pattern_args, args, copy))
                {
                    pattern_map.insert(begin(copy), end(copy));
                    return true;
                }
            }

            NGRAPH_DEBUG << "[MATCHER] Aborting at " << *graph_node << " for pattern "
                         << *pattern_node;
            return false;
        }

        bool Matcher::match(const Output<Node>& graph_value)
        {
            // clear our state
            m_matched_list.clear();
            return match(graph_value, PatternValueMap{});
        }

        bool Matcher::match(const Output<Node>& graph_value,
                            const PatternValueMap& previous_matches)
        {
            // clear our state
            m_match_root.reset();
            m_pattern_map.clear();

            // insert previous matches
            m_pattern_map.insert(previous_matches.cbegin(), previous_matches.cend());

            bool is_match = match_value(m_pattern_node, graph_value, m_pattern_map);
            if (is_match)
            {
                m_match_root = graph_value;
            }
            return is_match;
        }

        bool Matcher::match(const Output<Node>& graph_value, const PatternMap& previous_matches)
        {
            return match(graph_value, as_pattern_value_map(previous_matches));
        }

        namespace
        {
            std::set<Output<Node>>
                as_output_set(const std::set<std::shared_ptr<op::Label>>& label_set)
            {
                std::set<Output<Node>> result;
                for (auto label : label_set)
                {
                    result.insert(label);
                }
                return result;
            }
        }

        RecurrentMatcher::RecurrentMatcher(
            const Output<Node>& pattern,
            const Output<Node>& rpattern,
            const std::set<std::shared_ptr<op::Label>>& correlated_patterns)
            : RecurrentMatcher(pattern, rpattern, as_output_set(correlated_patterns))
        {
        }

        bool RecurrentMatcher::match(Output<Node> graph)
        {
            bool matched = false;
            Matcher m_initial(m_initial_pattern);
            Matcher m_repeat(m_pattern);
            Matcher& m = m_initial;
            PatternValueMap previous_matches;
            m_matches.clear();
            m_match_root = graph;

            // try to match one cell (i.e. pattern)
            while (m.match(graph, previous_matches))
            {
                matched = true;
                // move to the next cell
                graph = m.get_pattern_value_map()[m_recurrent_pattern];

                // copy bound nodes for the current pattern graph into a global matches map
                for (auto cur_match : m.get_pattern_value_map())
                {
                    m_matches[cur_match.first].push_back(cur_match.second);
                }

                // pre-populate the pattern map for the next cell with the bound nodes
                // from the current match. Only bound nodes whose labels are in
                // correlated_patterns are pre-populated. Skip other labels are
                // unbounded by default
                for (auto cor_pat : m_correlated_patterns)
                {
                    if (m.get_pattern_value_map().count(cor_pat) != 0)
                    {
                        // assert that bound nodes from the previous and current matches are the
                        // same
                        if (previous_matches.count(cor_pat) != 0)
                        {
                            if (previous_matches[cor_pat] != m.get_pattern_value_map()[cor_pat])
                            {
                                throw ngraph_error(
                                    "previous matches and current matches aren't consistent!");
                            }
                        }

                        previous_matches[cor_pat] = m.get_pattern_value_map()[cor_pat];
                    }
                }
                m = m_repeat;
            }

            if (!matched)
            {
                m_match_root.reset();
            }

            return matched;
        }
    }
}
