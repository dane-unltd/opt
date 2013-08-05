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

func (h *History) Update(m *Model) {
	if h.T != nil {
		h.T = append(h.T, m.time)
	}
	if h.X != nil {
		xt := make(mat.Vec, len(m.x))
		xt.Copy(m.x)
		h.X = append(h.X, xt)
	}
	if h.Grad != nil {
		if m.gradX != nil {
			gt := make(mat.Vec, len(m.gradX))
			gt.Copy(m.gradX)
			h.Grad = append(h.Grad, gt)
		}
	}
	if h.Obj != nil {
		h.Obj = append(h.Obj, m.objX)
	}
}
