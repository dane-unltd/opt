package linesearch

//Inexact line search using Armijo's rule.
type InexactSolver struct {
}

func (s InexactSolver) Solve(obj, grad Siso, f0, g0, xStart float64) (xNew float64, fNew float64) {
	beta := 0.5

	xNew = xStart
	fNew = obj(xNew)

	if fNew-f0 > 0.5*g0*xNew {
		fPrev := fNew
		xNew *= beta
		for {
			fNew = obj(xNew)
			if fNew-f0 <= 0.5*g0*xNew {
				if fPrev < fNew {
					xNew /= beta
					fNew = fPrev
				}
				break
			}
			fPrev = fNew
			xNew *= beta
		}
	} else {
		fPrev := fNew
		xNew /= beta
		for {
			fNew = obj(xNew)
			if fNew-f0 > 0.5*g0*xNew {
				if fPrev < fNew {
					xNew *= beta
					fNew = fPrev
				}
				break
			}
			fPrev = fNew
			xNew /= beta
		}
	}
	return
}
