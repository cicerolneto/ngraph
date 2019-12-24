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

#pragma once

#include <algorithm>
#include <functional>
#include <memory.h>

#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"

namespace ngraph
{
    namespace pass
    {
        class GraphRewrite;
    }

    namespace pattern
    {
        /// \brief Matcher matches (compares) two graphs
        ///
        class NGRAPH_API Matcher
        {
        public:
            using PatternMap = ngraph::pattern::PatternMap;

            // Avoid implicit string construction from nullptr.
            Matcher(const std::shared_ptr<Node>& pattern_node, std::nullptr_t name) = delete;

            Matcher(const Output<Node>& pattern_node)
                : m_pattern_node{pattern_node}
                , m_depth{0}
                , m_name{"Unnamed"}
                , m_strict_mode{false}
            {
            }

            Matcher(const Output<Node>& pattern_node, const std::string& name)
                : m_pattern_node(pattern_node)
                , m_depth{0}
                , m_name{name}
                , m_strict_mode{false}
            {
            }
            /// \brief Constructs a Matcher object
            ///
            /// \param pattern_node is a pattern sub graph that will be matched against input graphs
            /// \param name is a string which is used for logging and disabling a matcher
            /// \param strict_mode forces a matcher to consider shapes and ET of nodes
            Matcher(const Output<Node>& pattern_node, const std::string& name, bool strict_mode)
                : m_pattern_node(pattern_node)
                , m_depth(0)
                , m_name(name)
                , m_strict_mode(strict_mode)
            {
            }

            virtual ~Matcher() {}
            /// \brief Matches a pattern to \p graph_node
            ///
            /// \param graph_value is an input graph to be matched against
            bool match(const Output<Node>& graph_value);

            /// \brief Matches a pattern to \p graph_node
            ///
            /// \param graph_value is an input graph to be matched against
            /// \param previous_matches contains previous mappings from labels to nodes to use
            bool match(const Output<Node>& graph_value, const PatternMap& previous_matches);
            bool match(const Output<Node>& graph_value, const PatternValueMap& previous_matches);

            template <typename T>
            static std::shared_ptr<T> unique_match(std::shared_ptr<Node> node)
            {
                std::shared_ptr<T> matched;
                for (auto arg : node->get_arguments())
                {
                    if (auto t_casted = as_type_ptr<T>(arg))
                    {
                        if (matched)
                        {
                            throw ngraph_error("There's more than two arguments of the same type");
                        }
                        else
                        {
                            matched = t_casted;
                        }
                    }
                }
                return matched;
            }

            bool is_contained_match(const NodeVector& exclusions = {}, bool ignore_unused = true);
            const NodeVector get_matched_nodes() { return as_node_vector(m_matched_list); }
            const OutputVector& get_matched_values() { return m_matched_list; }
            void reset() {}
            const std::string& get_name() { return m_name; }
            std::shared_ptr<Node> get_pattern() { return m_pattern_node.as_single_output_node(); }
            Output<Node> get_pattern_value() { return m_pattern_node; }
            std::shared_ptr<Node> get_match_root();
            Output<Node> get_match_value();
            PatternMap get_pattern_map() const;
            PatternValueMap get_pattern_value_map() { return m_pattern_map; }
            /// \brief Low-level helper to match recurring patterns
            ///
            /// \param graph is a graph to be matched against
            /// \param pattern is a recurring pattern
            /// \param rpattern specifies a node to recur from next
            /// \param patterns a map from labels to matches
            friend op::Label; // TODO: refine to match_class

            size_t add_node(Output<Node> node);
            bool abort_match(size_t watermark, bool matched);

            bool virtual match_value(const ngraph::Output<Node>& pattern_value,
                                     const ngraph::Output<Node>& graph_value,
                                     PatternValueMap& pattern_map);

            bool is_strict_mode() { return m_strict_mode; }
            virtual bool match_arguments(const Output<Node>& pattern_value,
                                         const Output<Node>& graph_value,
                                         PatternValueMap& pattern_map);

            Output<Node> m_match_root;
            Output<Node> m_pattern_node;
            PatternValueMap m_pattern_map;
            OutputVector m_matched_list;

        private:
            static std::string pad(size_t num) { return std::string(num, ' '); }
            bool match_permutation(const OutputVector& pattern_args,
                                   const OutputVector& args,
                                   PatternValueMap& pattern_map);

            size_t m_depth;
            std::string m_name;
            bool m_strict_mode;
            bool m_follow_goe{false};
        };

        class RecurrentMatcher
        {
        public:
            /// \brief Constructs a RecurrentMatcher object. Reccurent Matchers are used to match
            ///        repeating patterns (e.g. RNN, LSTM, GRU cells)
            ///
            /// \param initial_pattern is a pattern sub graph describing the initial cell
            /// \param pattern is a pattern sub graph describing an individual cell
            /// \param rpattern is a (recurring) label to denote which node the next match should
            ///                 start at
            /// \param correlated_patterns is a set of labels whose bound nodes must remain the same
            ///                            across all cells
            RecurrentMatcher(const Output<Node>& initial_pattern,
                             const Output<Node>& pattern,
                             const Output<Node>& rpattern,
                             const std::set<Output<Node>>& correlated_patterns)
                : m_initial_pattern(initial_pattern)
                , m_pattern(pattern)
                , m_recurrent_pattern(rpattern)
                , m_correlated_patterns(correlated_patterns)
            {
            }

            /// \brief Constructs a RecurrentMatcher object. Reccurent Matchers are used to match
            ///        repeating patterns (e.g. RNN, LSTM, GRU cells)
            ///
            /// \param pattern is a pattern sub graph describing an individual cell
            /// \param rpattern is a (recurring) label to denote which node the next match should
            ///                 start at
            /// \param correlated_patterns is a set of labels whose bound nodes must remain the same
            ///                            across all cells
            RecurrentMatcher(const Output<Node>& pattern,
                             const Output<Node>& rpattern,
                             const std::set<Output<Node>>& correlated_patterns)
                : RecurrentMatcher(pattern, pattern, rpattern, correlated_patterns)
            {
            }

            RecurrentMatcher(const Output<Node>& pattern,
                             const Output<Node>& rpattern,
                             const std::set<std::shared_ptr<op::Label>>& correlated_patterns);

            /// \brief Returns a vector of bound nodes for a given label (used in a pattern
            /// describing an individual cell
            NodeVector get_bound_nodes_for_pattern(const Output<Node>& pattern) const
            {
                if (m_matches.count(pattern) == 0)
                {
                    throw ngraph_error("No bound nodes for a given label");
                }

                return as_node_vector(m_matches.at(pattern));
            }

            size_t get_number_of_recurrent_matches() const
            {
                if (m_matches.size() == 0)
                {
                    return 0;
                }

                return (*m_matches.begin()).second.size();
            }

            size_t get_number_of_bound_labels() const { return m_matches.size(); }
            /// \brief Tries to match a pattern for an individual cell to a given \p graph
            bool match(Output<Node> graph);

            std::shared_ptr<Node> get_match_root() { return m_match_root.get_node_shared_ptr(); }
            Output<Node> get_match_value() { return m_match_root; }
        private:
            Output<Node> m_initial_pattern;
            Output<Node> m_pattern;
            Output<Node> m_recurrent_pattern;
            const std::set<Output<Node>> m_correlated_patterns;
            RPatternValueMap m_matches;
            Output<Node> m_match_root;
        };
    }
}
