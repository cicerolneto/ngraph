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

#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

bool pattern::op::AnyOf::match_node(Matcher& matcher,
                                       const std::shared_ptr<Node>& graph_node,
                                       PatternMap& pattern_map)
{
    if (m_predicate(graph_node))
    {
        for (auto arg : graph_node->get_arguments())
        {
            PatternMap copy{pattern_map};
            if (matcher.match_node(get_argument(0), arg, copy))
            {
                pattern_map.insert(begin(copy), end(copy));
                return true;
            }
        }
    }
    return false;
}
