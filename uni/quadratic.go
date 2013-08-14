package uni

import (
	"math"
)

//Exact line search for strictly quasi-convex functions
type Quadratic struct{}

func NewQuadratic() *Quadratic {
	return &Quadratic{}
}

func (sol *Quadratic) Solve(o Function, in *Solution, p *Params) *Result {
	r := NewResult(in)
	obj := ObjWrapper{r: r, o: o}
	r.init(obj)
	h := NewHelper(r.Solution)

	var eps float64

	fNew := 0.0
	xNew := 0.0

	if math.IsInf(r.UB, 1) {
		xNew = r.X
		fNew = r.ObjX
		if math.IsNaN(xNew) {
			xNew = 1
			fNew = obj.Val(xNew)
		}
		if math.IsNaN(fNew) {
			fNew = obj.Val(xNew)
		}

		step := r.X - r.LB
		if fNew < r.ObjLB {
			r.ObjX = fNew
			r.X = xNew

			r.UB = r.X + step
			r.ObjUB = obj.Val(r.UB)
			for r.ObjUB <= r.ObjX {
				r.LB = r.X
				r.ObjLB = r.ObjX

				r.X = r.UB
				r.ObjX = r.ObjUB

				step *= 2
				r.UB = r.X + step
				r.ObjUB = obj.Val(r.UB)

				if h.update(r, p); r.Status != 0 {
					return r
				}
			}
		} else {
			r.ObjUB = fNew
			r.UB = xNew

			step *= 0.5

			r.X = r.LB + step
			r.ObjX = obj.Val(r.X)
			for r.ObjX >= r.ObjLB {
				r.UB = r.X
				r.ObjUB = r.ObjX

				step *= 0.5
				r.X = r.LB + step
				r.ObjX = obj.Val(r.X)

				if h.update(r, p); r.Status != 0 {
					return r
				}
			}
		}
	} else {
		eps = math.Min(p.XTolAbs, p.XTolRel*h.initialInterval)
		if eps <= 0 {
			eps = 0.01 * h.initialInterval
		}
		r.UB = r.UB
		r.ObjUB = r.ObjUB
		if math.IsNaN(r.ObjUB) {
			r.ObjUB = obj.Val(r.UB)
		}
		if r.ObjUB < r.ObjLB {
			r.X = r.UB - eps
			r.ObjX = obj.Val(r.X)
			if r.ObjX >= r.ObjUB {
				r.X = r.UB
				r.ObjX = r.ObjUB
				r.Status = XRelConv
				return r
			}
		} else {
			r.X = 0.5 * r.UB
			r.ObjX = obj.Val(r.X)
			for r.ObjX >= r.ObjLB {
				r.X *= 0.5
				r.ObjX = obj.Val(r.X)

				if h.update(r, p); r.Status != 0 {
					return r
				}
			}
		}
	}

	if eps == 0 {
		h.initialInterval = r.UB - r.LB
		eps = math.Min(p.XTolAbs, p.XTolRel*h.initialInterval)
		if eps == 0 {
			eps = 0.01 * h.initialInterval
		}
	}

	for {
		//optimum of quadratic fit
		xNew = -0.5 * (r.UB*r.UB*(r.ObjLB-r.ObjX) + r.X*r.X*(r.ObjUB-r.ObjLB) + r.LB*r.LB*(r.ObjX-r.ObjUB)) /
			(r.UB*(r.ObjX-r.ObjLB) + r.X*(r.ObjLB-r.ObjUB) + r.LB*(r.ObjUB-r.ObjX))

		if math.Abs(r.LB-xNew) < 0.4*eps {
			xNew = r.LB + 0.4*eps
		}
		if math.Abs(r.UB-xNew) < 0.4*eps {
			xNew = r.UB - 0.4*eps
		}
		if math.Abs(r.X-xNew) < 0.4*eps {
			if r.UB-r.X > r.X-r.LB {
				xNew = r.X + 0.4*eps
			} else {
				xNew = r.X - 0.4*eps
			}
		}

		fNew = obj.Val(xNew)

		if !(xNew > r.LB && xNew < r.UB) || (xNew < r.X && fNew > r.ObjLB) ||
			(xNew > r.X && fNew > r.ObjUB) {
			if r.UB-r.X > r.X-r.LB {
				xNew = (r.X + r.UB) / 2
			} else {
				xNew = (r.X + r.LB) / 2
			}
			fNew = obj.Val(xNew)
		}

		if xNew > r.X {
			if fNew >= r.ObjX {
				r.UB = xNew
				r.ObjUB = fNew
			} else {
				r.LB = r.X
				r.ObjLB = r.ObjX
				r.X = xNew
				r.ObjX = fNew
			}
		} else {
			if fNew >= r.ObjX {
				r.LB = xNew
				r.ObjLB = fNew
			} else {
				r.UB = r.X
				r.ObjUB = r.ObjX
				r.X = xNew
				r.ObjX = fNew
			}
		}

		if h.update(r, p); r.Status != 0 {
			break
		}
	}
	return r
}
