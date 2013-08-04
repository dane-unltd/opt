package uni

type Cubic struct {
	Tol     float64
	IterMax int
	// Tunable parameters
	stepDecreaseMin float64 // Minimum allowable decrease (must be a number between [0,1)) default 0.0001
	stepDecreaseMax float64 // When decreasing what is the high
	stepIncreaseMin float64
	stepIncreaseMax float64
}

func (sol *Cubic) Solve(m *Model) error {
	/*
		if math.IsNaN(f0) {
			f0 = obj(x0)
		}
		if math.IsNaN(g0) {
			g0 = grad(x0)
		}

		delta := 0.0
		posDir := true
		negGrad := g0 < 0

		step := 1
		g := g0
		f := f0

		var stepMultiplier float64
		updateCurrPoint := false
		reverseDirection := false
		var trialX, trialF, trialG float64
		var newStepSize float64

		iter := 1
		for ; iter <= sol.IterMax; iter++ {
			if negGrad {
				trialX = x0 + step
			} else {
				trialX = x0 - step
			}
			trialF = obj(trialX)
			trialG = grad(trialX)

			absTrialG := math.Abs(trialG)
			// Find guess for next point
			deltaF := trialF - f
			decrease := (deltaF <= 0)

			// See if we can trust the deltaF
			// measurement
			var canTrustDeltaF bool
			divisor := math.Max(math.Abs(trialF), math.Abs(currF))
			if divisor == 0 {
				canTrustDeltaF = true // Both are zero, so is >= 0
			} else {
				if math.Abs(deltaF) > divisor*eps { // Change large enough to trust
					canTrustDeltaF = true
				}
				//otherwise can't trust
			}
			changeInDerivSign := (currG > 0 && trialG < 0) || (currG < 0 && trialG > 0)
			decreaseInDerivMagnitude := (absTrialG < math.Abs(currG))
			// Find coefficients of the cubic polynomial fit between the
			// current point and the new point
			// Derived from fitting a cubic between (0, CurrF) and
			// (1,TrialF).
			// Apply transformations later to reshift the
			// coordinate axis

			// Need to play games with derivatives
			trialFitG := trialG
			currFitG := currG
			if cubic.initialGradNegative {
				trialFitG *= -1
				currFitG *= -1
			}
			if cubic.currStepDirectionPositive {
				trialFitG *= -1
				currFitG *= -1
			}
			var a, b, c float64
			a = trialG + currG - 2*deltaF
			b = 3*deltaF - 2*currG - trialG

			c = currG
			det := (math.Pow(b, 2) - 3*a*c)

		}
	*/
	return nil
}
