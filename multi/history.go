package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"time"
)

type History struct {
	T    []time.Duration
	X    []mat.Vec
	Obj  []float64
	Grad []mat.Vec
}

func (h *History) Update(m *Result) Status {
	if h.T != nil {
		h.T = append(h.T, m.Time)
	}
	if h.X != nil {
		xt := make(mat.Vec, len(m.X))
		xt.Copy(m.X)
		h.X = append(h.X, xt)
	}
	if h.Grad != nil {
		if m.GradX != nil {
			gt := make(mat.Vec, len(m.GradX))
			gt.Copy(m.GradX)
			h.Grad = append(h.Grad, gt)
		}
	}
	if h.Obj != nil {
		h.Obj = append(h.Obj, m.ObjX)
	}
	return NotTerminated
}
