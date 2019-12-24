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

#include <functional>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            class Label;
        }

        class Matcher;
        class MatchState;

        using RPatternValueMap = std::map<Output<Node>, OutputVector>;
        using PatternValueMap = std::map<Output<Node>, Output<Node>>;

        using PatternMap = std::map<Output<Node>, std::shared_ptr<Node>>;

        PatternMap as_pattern_map(const PatternValueMap& pattern_value_map);
        PatternValueMap as_pattern_value_map(const PatternMap& pattern_map);

        template <typename T>
        std::function<bool(std::shared_ptr<Node>)> has_class()
        {
            auto pred = [](std::shared_ptr<Node> node) -> bool { return is_type<T>(node); };

            return pred;
        }

        namespace op
        {
            using NodePredicate = std::function<bool(std::shared_ptr<Node>)>;
            using ValuePredicate = std::function<bool(const Output<Node>& value)>;

            ValuePredicate as_value_predicate(NodePredicate pred);

            class NGRAPH_API Pattern : public Node
            {
            public:
                /// \brief \p a base class for \sa Skip and \sa Label
                ///
                Pattern(const OutputVector& wrapped_values, ValuePredicate pred)
                    : Node(wrapped_values)
                    , m_predicate(pred)
                {
                    if (!m_predicate)
                    {
                        m_predicate = [](const Output<Node>&) { return true; };
                    }
                }

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& /* new_args */) const override
                {
                    throw ngraph_error("Uncopyable");
                }

                ValuePredicate get_predicate() const;

                bool is_pattern() const override { return true; }
                virtual bool match_value(pattern::Matcher& matcher,
                                         const Output<Node>& pattern_value,
                                         const Output<Node>& graph_value,
                                         PatternValueMap& pattern_map) = 0;

            protected:
                ValuePredicate m_predicate;
            };
        }
    }
}
