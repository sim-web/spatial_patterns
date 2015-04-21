from tables import IsDescription, Int32Col, Float64Col, StringCol, BoolCol

name_size = 32
coord_size = 256
eqs_size = 256
units_size = 12

class Quantity(IsDescription):
    value = Float64Col()
    units = StringCol(units_size)

class VariantType(IsDescription):
    # vartype can be one of: strcode, float, integer, array, table
    vartype = StringCol(name_size)
    # array, strcode, and table need to store a string in varstring
    #   - strcode is the case where executable code is specified in place of a value
    #   - table stores the name of the referenced table group
    #   - array stores the name of the referenced array
    varstring = StringCol(eqs_size)
    # Both arrays and single values have units
    varint = Int32Col()
    varflt = Float64Col()
    units = StringCol(units_size)

class NamedVariantType(IsDescription):
    name = StringCol(name_size)
    vartype = VariantType()

class TimedVariantType(IsDescription):
    namedvartype = NamedVariantType()
    dt = Quantity()

class Population(IsDescription):
    name = StringCol(name_size)
    model = StringCol(name_size)
    size = Int32Col()
    
class Subpopulation(IsDescription):
    super = StringCol(name_size)
    name = StringCol(name_size)
    size = Int32Col()

class Connection(IsDescription):
    name = StringCol(name_size)
    popname_pre = StringCol(name_size)
    popname_post = StringCol(name_size)
    synapse = StringCol(name_size)
    connectivity = VariantType()
    delays = VariantType()

class LinkedRanges(IsDescription):
    coord_a = StringCol(coord_size)
    coord_b = StringCol(coord_size)

class ParamSpaceCoordinate(IsDescription):
    coord = StringCol(coord_size)
    units = StringCol(units_size)
    column = Int32Col()
    alias = BoolCol()

class AliasedParameters(IsDescription):
    name = StringCol(name_size)
    issparse = BoolCol()
