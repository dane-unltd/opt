package uni

import (
	"math"
	"time"
)

//Exact line search for strictly quasi-convex functions
type Quadratic struct {
	Termination
}

func NewQuadratic() *Quadratic {
	return &Quadratic{
		Termination: Termination{
			IterMax: 100,
			TimeMax: time.Minute,
		},
	}
}

func (sol *Quadratic) Solve(o Function, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := ObjWrapper{r: r, o: o}
	r.init(obj)

	upd = append(upd, sol.Termination)

	initialTime := time.Now()

	fNew := 0.0
	xNew := 0.0

	eps := 0.0

	if math.IsInf(r.XUpper, 1) {
		xNew = r.X
		fNew = r.Obj
		if math.IsNaN(xNew) {
			xNew = 1
			fNew = obj.Val(xNew)
		}
		if math.IsNaN(fNew) {
			fNew = obj.Val(xNew)
		}

		step := r.X - r.XLower
		if fNew < r.ObjLower {
			r.Obj = fNew
			r.X = xNew

			r.XUpper = r.X + step
			r.ObjUpper = obj.Val(r.XUpper)
			for r.ObjUpper <= r.Obj {
				r.XLower = r.X
				r.ObjLower = r.Obj

				r.X = r.XUpper
				r.Obj = r.ObjUpper

				step *= 2
				r.XUpper = r.X + step
				r.ObjUpper = obj.Val(r.XUpper)

				if doUpdates(r, initialTime, upd) != 0 {
					return r
				}
			}
		} else {
			r.ObjUpper = fNew
			r.XUpper = xNew

			step *= 0.5

			r.X = r.XLower + step
			r.Obj = obj.Val(r.X)
			for r.Obj >= r.ObjLower {
				r.XUpper = r.X
				r.ObjUpper = r.Obj

				step *= 0.5
				r.X = r.XLower + step
				r.Obj = obj.Val(r.X)

				if doUpdates(r, initialTime, upd) != 0 {
					return r
				}
			}
		}
	} else {

		r.XUpper = r.XUpper
		r.ObjUpper = r.ObjUpper
		if math.IsNaN(r.ObjUpper) {
			r.ObjUpper = obj.Val(r.XUpper)
		}
		if r.ObjUpper < r.ObjLower {
			for r.ObjUpper < r.ObjLower {
				eps = 0.01 * (r.XUpper - r.XLower)
				r.X = r.XUpper - eps
				r.Obj = obj.Val(r.X)
				if r.Obj >= r.ObjUpper {
					r.XLower = r.X
					r.ObjLower = r.Obj
				} else {
					break
				}
				if doUpdates(r, initialTime, upd) != 0 {
					return r
				}
			}
		} else {

			r.X = 0.5 * r.XUpper
			r.Obj = obj.Val(r.X)
			for r.Obj >= r.ObjLower {
				r.X *= 0.5
				r.Obj = obj.Val(r.X)

				if doUpdates(r, initialTime, upd) != 0 {
					return r
				}
			}
		}
	}

	for {
		eps = 0.01 * (r.XUpper - r.XLower)

		//optimum of quadratic fit
		xNew = -0.5 * (r.XUpper*r.XUpper*(r.ObjLower-r.Obj) + r.X*r.X*(r.ObjUpper-r.ObjLower) + r.XLower*r.XLower*(r.Obj-r.ObjUpper)) /
			(r.XUpper*(r.Obj-r.ObjLower) + r.X*(r.ObjLower-r.ObjUpper) + r.XLower*(r.ObjUpper-r.Obj))

		if math.Abs(r.XLower-xNew) < eps {
			xNew = r.XLower + eps
		}
		if math.Abs(r.XUpper-xNew) < eps {
			xNew = r.XUpper - eps
		}
		if math.Abs(r.X-xNew) < eps {
			if r.XUpper-r.X > r.X-r.XLower {
				xNew = r.X + eps
			} else {
				xNew = r.X - eps
			}
		}

		fNew = obj.Val(xNew)

		if !(xNew > r.XLower && xNew < r.XUpper) || (xNew < r.X && fNew > r.ObjLower) ||
			(xNew > r.X && fNew > r.ObjUpper) {
			if r.XUpper-r.X > r.X-r.XLower {
				xNew = (r.X + r.XUpper) / 2
			} else {
				xNew = (r.X + r.XLower) / 2
			}
			fNew = obj.Val(xNew)
		}

		if xNew > r.X {
			if fNew >= r.Obj {
				r.XUpper = xNew
				r.ObjUpper = fNew
			} else {
				r.XLower = r.X
				r.ObjLower = r.Obj
				r.X = xNew
				r.Obj = fNew
			}
		} else {
			if fNew >= r.Obj {
				r.XLower = xNew
				r.ObjLower = fNew
			} else {
				r.XUpper = r.X
				r.ObjUpper = r.Obj
				r.X = xNew
				r.Obj = fNew
			}
		}

		if doUpdates(r, initialTime, upd) != 0 {
			return r
		}
	}
	return r
}
