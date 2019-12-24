#include "ngraph/pattern/op/or.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo pattern::op::Or::type_info;

const NodeTypeInfo& pattern::op::Or::get_type_info() const
{
    return type_info;
}

bool pattern::op::Or::match_value(Matcher& matcher,
                                  const Output<Node>& pattern_value,
                                  const Output<Node>& graph_value,
                                  PatternValueMap& pattern_map)
{
    for (auto input_value : input_values())
    {
        PatternValueMap copy(pattern_map);
        if (m_predicate(input_value) && matcher.match_value(input_value, graph_value, copy))
        {
            swap(pattern_map, copy);
            return true;
        }
    }
    return false;
}
