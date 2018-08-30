# coding=utf-8

from . import session
from . import graph
from . import ops
from . import nn
from . import train
from . import models

Session = session.Session
Graph = graph.Graph

# alias for convenience
constant    =       ops.Constant
placeholder =       ops.Placeholder
Variable    =       ops.Variable
add         =       ops.AddOp
minus       = sub = ops.SubOp
multiple    = mul = ops.MulOp
matmul      =       ops.MatmulOp
divide      = div = ops.DivOp
power       = pow = ops.PowOp

