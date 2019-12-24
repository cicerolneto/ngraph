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
            /// \brief Ors are used to allow on of several patterns
            class NGRAPH_API Or : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternOr", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief creates a Any node containing a sub-pattern described by \sa type and \sa
                ///        shape.
                Or(const OutputVector& wrapped_values, ValuePredicate pred)
                    : Pattern(wrapped_values, pred)
                {
                }

                Or(const OutputVector& wrapped_values)
                    : Or(wrapped_values, [](const Output<Node>&) { return true; })
                {
                }

                bool match_value(pattern::Matcher& matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value,
                                 PatternValueMap& pattern_map) override;
            };
        }
    }
}
