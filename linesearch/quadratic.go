package linesearch

//Exact line search for strictly quasi-convex functions
type Quadratic struct {
	Tol float64
}

func (s Quadratic) Solve(obj, grad Siso, f0, g0, xStart float64) (xNew float64, fNew float64) {
	x1 := 0.0
	f1 := f0

	f2, f3 := 0.0, 0.0
	x2, x3 := 0.0, 0.0

	xNew = xStart
	fNew = obj(xNew)

	if fNew < f1 {
		f2 = fNew
		x2 = xNew

		x3 = 2 * x2
		f3 = obj(x3)
		for f3 <= f2 {
			x3 *= 2
			f3 = obj(x3)

		}
	} else {
		f3 = fNew
		x3 = xNew

		x2 = 0.5 * x3
		f2 = obj(x2)
		for f2 >= f1 {
			x2 *= 0.5
			f2 = obj(x2)
		}
	}

	for x3-x1 > s.Tol*xStart {
		xNew = -0.5 * (x3*x3*(f1-f2) + x2*x2*(f3-f1) + x1*x1*(f2-f3)) /
			(x3*(f2-f1) + x2*(f1-f3) + x1*(f3-f2))
		fNew = obj(xNew)
		if xNew > x2 {
			if fNew >= f2 {
				x3 = xNew
			} else {
				x1 = x2
				x2 = xNew
				f2 = fNew
			}
		} else {
			if fNew >= f2 {
				x1 = xNew
			} else {
				x3 = x2
				x2 = xNew
				f2 = fNew
			}
		}
	}
	return
}
