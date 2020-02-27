echo "Current working directory: \n"
echo $PWD

path27=$(which python2.7)
path3=$(which python3)

if [ test -z "$path" ]
then
      echo "\$python2.7 is missing. Please install python 2.7"
      exit
else
      echo "python2.7 available.\n"
fi

if [ test -z "$path3" ]
then
      echo "\$python3 is missing. Please install python3"
      exit
else
      echo "python3 available.\n"
fi

while getopts "d:" opt; do
    case $opt in
    a) dist=$OPTARG ;;
    \?) ;; # Handle error: unknown option or missing required argument.
    esac
done

if [ test -z "$dist" ]
then
    echo "No distribution specified"
    exit
fi

echo "Analysing the state space.. \nIt trains a Tensorflow model and produces plots.."

python3 execute.py --distribution "$dist"

if [ $dist -eq "beta" ]
then

    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_beta.py --dist tmp_models/beta.py --units tmp_models/beta_units.py --scores tmp_models/beta_scores.py

elif [ $dist -eq "cauchy" ]
then

    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_cauchy.py --dist tmp_models/cauchy.py --units tmp_models/cauchy_units.py --scores tmp_models/cauchy_scores.py

elif [ $dist -eq "gamma" ]
then

    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_gamma.py --dist tmp_models/gamma.py --units tmp_models/gamma_units.py --scores tmp_models/gamma_scores.py

elif [ $dist -eq "rayleigh" ]
then

    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_rayleigh.py --dist tmp_models/rayleigh.py --units tmp_models/rayleigh_units.py --scores tmp_models/rayleigh_scores.py

elif [ $dist -eq "weibull" ]
then

    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_weibull.py --dist tmp_models/weibull.py --units tmp_models/weibull_units.py --scores tmp_models/weibull_scores.py

elif [ $dist -eq "all" ]
then
    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_beta.py --dist tmp_models/beta.py --units tmp_models/beta_units.py --scores tmp_models/beta_scores.py
    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_cauchy.py --dist tmp_models/cauchy.py --units tmp_models/cauchy_units.py --scores tmp_models/cauchy_scores.py
    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_gamma.py --dist tmp_models/gamma.py --units tmp_models/gamma_units.py --scores tmp_models/gamma_scores.py
    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_rayleigh.py --dist tmp_models/rayleigh.py --units tmp_models/rayleigh_units.py --scores tmp_models/rayleigh_scores.py
    python2.7 ../../qsr_lib_ocr/qsr_lib/scripts/qstag_example_weibull.py --dist tmp_models/weibull.py --units tmp_models/weibull_units.py --scores tmp_models/weibull_scores.py
else
    echo "No distribution specified.\n"
    exit
fi