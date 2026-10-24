"""Physical and reporting constants.

Every number here is either a physical constant, a published default, or an
explicit assumption. Assumptions are flagged so they are easy to find and change.
"""

G = 9.81  # m/s^2

# Share of the truck's weight on the driven axles. With the tyre-road friction of the
# surface this sets the largest force the wheels can transmit (matters in monsoon mud).
DRIVE_AXLE_LOAD_SHARE = 0.6

# --- Diesel properties -----------------------------------------------------
# Density: EN 590 allows 820-845 kg/m^3 at 15 C; Indian BS-VI diesel is in a
# similar band. 0.84 is a mid-range value.
DIESEL_DENSITY_KG_PER_L = 0.84
# Net calorific value, IPCC 2006 default for gas/diesel oil: 43.0 TJ/Gg (= MJ/kg).
DIESEL_LHV_MJ_PER_KG = 43.0

# --- CO2 factor ------------------------------------------------------------
# IPCC 2006 Guidelines default CO2 emission factor for gas/diesel oil is
# 74.1 t CO2/TJ (as reproduced in Annex VI of EU Implementing Regulation
# 2018/2066). 74.1 g/MJ x 43.0 MJ/kg x 0.84 kg/L = 2.676 kg CO2 per litre.
# An independent carbon-balance estimate (86 % carbon by weight, 0.85 kg/L,
# 44/12) also gives 2.68. See README "Emission factor".
CO2_KG_PER_L_DIESEL = 2.68
# The factor above covers CO2 only. CH4 and N2O from the tailpipe are not
# included, so "CO2e" in the reports is a CO2-only figure. Change this constant
# (or pass a factor to the metrics functions) to use a CO2e factor instead.

# --- Economics (ASSUMPTIONS, configurable) ----------------------------------
# Used only to turn the time-versus-fuel trade-off into one number for the
# planner and to report fuel cost in rupees. Adjust to current prices.
FUEL_PRICE_INR_PER_L = 90.0
TIME_VALUE_INR_PER_H = 800.0  # driver + truck time cost while hauling

# Planner weight on time, in litres of diesel per second of travel time.
TIME_WEIGHT_L_PER_S = (TIME_VALUE_INR_PER_H / FUEL_PRICE_INR_PER_L) / 3600.0
