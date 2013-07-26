package unc

import (
	"github.com/dane-unltd/linalg/mat"
)

type Miso func(x mat.Vec) float64
type Mimo func(in, out mat.Vec)

type Projection func(x mat.Vec)

type OptStatus int

const (
	OK OptStatus = iota
	MaxIter
)

type Result struct {
	Obj     float64
	ObjHist []float64
	Grad    mat.Vec
	Iter    int
	Status  OptStatus
}
