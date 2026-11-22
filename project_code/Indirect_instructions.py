tasks_ = [
    {
        "plant_crop_water_task": {
            "direct": [
                "Plant two crops and keep both watered.",
                "Plant two crops, then water them whenever necessary to prevent them from drying out.",
                "Establish two crops in the fields and ensure both receive sufficient water throughout their growth."
            ],
            "indirect": [
                "The farm needs two planted crops, and both should have enough water to grow.",
                "A pair of growing crops is needed, with neither plot allowed to dry out.",
                "The fields should end up with two healthy crops supported by regular watering."
            ],
            "implicit": [
                "Two healthy crops would be a good start, but dry plots could put their growth at risk.",
                "The farm is still short of two growing crops, and moisture is essential for them to thrive.",
                "A productive field needs a pair of established plants and soil that stays moist."
            ]
        }
    },

    {
        "harvest_crop_task": {
            "direct": [
                "Harvest three crops by the end of day 10.",
                "Collect the harvest from three crops before day 10 ends.",
                "Complete three crop harvests no later than the end of the tenth day."
            ],
            "indirect": [
                "The farm should have three harvested crops by the end of day 10.",
                "Three crops need to be brought in before the tenth day is over.",
                "By day 10, the fields should have yielded three harvested crops."
            ],
            "implicit": [
                "Day 10 is approaching, and the farm still needs three completed crop yields.",
                "Three crop yields need to be secured before the current deadline passes.",
                "The farm's first three yields must be ready before the tenth day is over."
            ]
        }
    },

    {
        "harvest_crop_earn_money_task": {
            "direct": [
                "Harvest four crops and sell produce to earn $150 by the end of day 18.",
                "Before day 18 ends, harvest four crops and generate $150 in sales revenue.",
                "Complete four crop harvests and earn $150 from selling produce no later than day 18."
            ],
            "indirect": [
                "The farm needs four harvested crops and $150 in sales revenue before the end of day 18.",
                "By day 18, four crops should be harvested and produce sales should bring in $150.",
                "The farm's records should show four harvests and $150 earned from sales by the eighteenth day."
            ],
            "implicit": [
                "Day 18 is approaching, and both the farm's harvest output and sales income need attention.",
                "Four completed harvests and a $150 return from produce would put the farm on track before day 18.",
                "The farm's deadline is approaching; the silo needs to contribute $150 in sales, alongside four completed crop yields."
            ]
        }
    }

]
tasks = [
    {
        "plant_crop_water_task": {
            "direct": [
                "Plant two crops and water them.",
                "Plant two crops and give water to each of them.",
                "Plant two crops in the farm plots and water both of them."
            ],
            "indirect": [
                "The farm needs two planted crops, and both should have enough water to grow.",
                "A pair of growing crops is needed, with both plots receiving water.",
                "The plots should end up with two crops in the ground that have been given water."
            ],
            "implicit": [
                "Two farm plots are empty, and no rain is on the way, so anything growing there will struggle without help from us.",
                "We have a standing request for two newly established crops, but the rovers are out of charge, so the plots will not look after themselves.",
                "Two plots sit empty and bone-dry, and the farm wants something alive and thriving in each of them this season."
            ]
        }
    },

    {
        "harvest_crop_task": {
            "direct": [
                "Harvest three crops by the end of day 10.",
                "Collect the harvest from three crops before day 10 ends.",
                "Complete three crop harvests no later than the end of the tenth day."
            ],
            "indirect": [
                "The farm should have three harvested crops by the end of day 10.",
                "Three crops need to be brought in before the end of the tenth day.",
                "By the end of day 10, the plots should have yielded three harvested crops."
            ],
            "implicit": [
                "Day 10 is the deadline on the objective. The plots are bare and the silo is empty, and the farm is expected to have produce from three crops in the silo by then.",
                "A frost is due at the end of day 10. The plots are bare and the silo is empty, yet the farm is expected to hold produce from three crops by then.",
                "By the time day 10 ends, the silo should hold produce from three crops. At the moment the plots are bare and the silo holds nothing."
            ]
        }
    },

    {
        "harvest_crop_earn_money_task": {
            "direct": [
                "Harvest four crops and earn $150 from sales by the end of day 18.",
                "Before the eighteenth day ends, harvest four crops and generate $150 in sales revenue.",
                "Earn $150 from sales and harvest four crops before the end of day 18."
            ],
            "indirect": [
                "The farm needs four harvested crops and $150 in sales revenue before the end of day 18.",
                "By the end of day 18, four crops should be harvested and produce sales should bring in $150.",
                "The farm's records should show four harvests and $150 earned from sales by the end of the eighteenth day."
            ],
            "implicit": [
                "The season review comes at the end of day 18: the plan calls for produce from four crops and $150 of income, and neither has been recorded yet.",
                "When day 18 ends, the objective will be checked for four crops' worth of produce and $150 in income. Right now, the silo is empty and no income is recorded.",
                "The market runs on daily prices, and by the end of day 18 the farm is expected to show produce from four crops and $150 in income. Neither exists yet."
            ]
        }
    }
]
