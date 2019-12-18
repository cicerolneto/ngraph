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

        using RPatternMap = std::map<std::shared_ptr<op::Label>, NodeVector>;
        using PatternMap = std::map<std::shared_ptr<op::Label>, std::shared_ptr<Node>>;

        template <typename T>
        std::function<bool(std::shared_ptr<Node>)> has_class()
        {
            auto pred = [](std::shared_ptr<Node> node) -> bool { return is_type<T>(node); };

            return pred;
        }

        namespace op
        {
            using Predicate = std::function<bool(std::shared_ptr<Node>)>;

            class NGRAPH_API Pattern : public Node
            {
            public:
                /// \brief \p a base class for \sa Skip and \sa Label
                ///
                Pattern(const OutputVector& wrapped_values, Predicate pred)
                    : Node(wrapped_values)
                    , m_predicate(pred)
                {
                    if (!m_predicate)
                    {
                        m_predicate = [](std::shared_ptr<Node>) { return true; };
                    }
                }

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& /* new_args */) const override
                {
                    throw ngraph_error("Uncopyable");
                }

                Predicate get_predicate() const;

                bool is_pattern() const override { return true; }
                virtual bool match_node(pattern::Matcher& matcher,
                                           const std::shared_ptr<Node>& graph_node,
                                           pattern::PatternMap& pattern_map) = 0;

            protected:
                std::function<bool(std::shared_ptr<Node>)> m_predicate;
            };
        }
    }
}
