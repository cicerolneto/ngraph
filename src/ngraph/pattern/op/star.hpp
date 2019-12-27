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

#include "ngraph/node.hpp"
#include "ngraph/pattern/op/pattern.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            /// \brief Stars are used to allow repeat patterns
            class NGRAPH_API Star : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternStar", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief Creates a Star pattern
                /// \param pattern the repeating pattern
                /// \param labels Labels where the repeat may occur
                Star(const OutputVector& exit, ValuePredicate pred)
                    : Pattern(exit, pred)
                {
                }

                Star(const OutputVector& exit)
                    : Star(exit, [](const Output<Node>&) { return true; })
                {
                }

                void set_repeat(const Output<Node>& repeat)
                {
                    m_repeat_node = repeat.get_node();
                    m_repeat_index = repeat.get_index();
                }

                Output<Node> get_repeat() const
                {
                    return m_repeat_node == nullptr
                               ? Output<Node>()
                               : Output<Node>{m_repeat_node->shared_from_this(), m_repeat_index};
                }

                bool match_value(pattern::Matcher& matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value,
                                 PatternValueMap& pattern_map) override;

            protected:
                Node* m_repeat_node{nullptr};
                size_t m_repeat_index{0};
            };
        }
    }
}
