package opt

import (
	. "github.com/dane-unltd/linalg/matrix"
)

type Objective struct {
	F func(x Vec) float64
	G func(x, g Vec)
}

type Projection func(x Vec)
