package opt

import (
	"github.com/dane-unltd/linalg/matrix"
)

type Miso func(x matrix.Vec) float64
type Mimo func(in, out matrix.Vec)

type Projection func(x matrix.Vec)

type OptStatus int

const (
	OK OptStatus = iota
	MaxIter
)

type Result struct {
	Obj    float64
	Grad   matrix.Vec
	Iter   int
	Status OptStatus
}
