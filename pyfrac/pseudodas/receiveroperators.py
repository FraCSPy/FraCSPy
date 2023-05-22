from pylops.basicoperators import *


def cubePDoperator(nr_x, nt):

    n_faces = 6
    # LINE A
    RLA_Op = Restriction(dims=[n_faces, nr_x, nr_x, nt],
                         iava=[0, 1, 2, 3],
                         axis=0
                         )
    SLA_Op = Sum((4, nr_x, nr_x, nt), axis=2)

    # LINE B
    RLB_Op = Restriction(dims=[n_faces, nr_x, nr_x, nt],
                         iava=[2, 3, 4, 5],
                         axis=0
                         )
    SLB_Op = Sum((4, nr_x, nr_x, nt), axis=2)

    # LINE C
    RLC_Op = Restriction(dims=[n_faces, nr_x, nr_x, nt],
                         iava=[0, 1, 4, 5],
                         axis=0
                         )
    SLC_Op = Sum((4, nr_x, nr_x, nt), axis=2)

    # COMBINING OPERATORS
    Vop = VStack([SLA_Op.H @ SLA_Op @ RLA_Op,
                  SLB_Op.H @ SLB_Op @ RLB_Op,
                  SLC_Op.H @ SLC_Op @ RLC_Op
                  ])

    return Vop
