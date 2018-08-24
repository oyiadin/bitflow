#coding=utf-8

from . import session
from . import graph
from . import ops
from . import train

Session = session.Session
Graph = graph.Graph


# alias for convenience
constant    =       ops.constant
placeholder =       ops.placeholder
Variable    =       ops.Variable
add         =       ops.AddOp
minus       = sub = ops.SubOp
multiple    = mul = ops.MulOp
matmul      =       ops.MatmulOp
divide      = div = ops.DivOp
power       = pow = ops.PowOp

