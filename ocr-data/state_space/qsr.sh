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

    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_beta.py --beta "$PWD"/ocr-data/state_speech/tmp_models/beta.npy --units "$PWD"/ocr-data/state_speech/tmp_models/beta_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/beta_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/beta_timestamp.npy

elif [ $dist -eq "cauchy" ]
then

    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_cauchy.py --cauchy "$PWD"/ocr-data/state_speech/tmp_models/cauchy.npy --units "$PWD"/ocr-data/state_speech/tmp_models/cauchy_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/cauchy_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/cauchy_timestamp.npy

elif [ $dist -eq "gamma" ]
then

    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_gamma.py --gamma "$PWD"/ocr-data/state_speech/tmp_models/gamma.npy --units "$PWD"/ocr-data/state_speech/tmp_models/gamma_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/gamma_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/gamma_timestamp.npy

elif [ $dist -eq "rayleigh" ]
then

    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_rayleigh.py --rayleigh "$PWD"/ocr-data/state_speech/tmp_models/rayleigh.npy --units "$PWD"/ocr-data/state_speech/tmp_models/rayleigh_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/rayleigh_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/rayleigh_timestamp.npy

elif [ $dist -eq "weibull" ]
then

    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_weibull.py --weibull "$PWD"/ocr-data/state_speech/tmp_models/weibull.npy --units "$PWD"/ocr-data/state_speech/tmp_models/weibull_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/weibull_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/weibull_timestamp.npy

elif [ $dist -eq "all" ]
then
    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_beta.py --beta "$PWD"/ocr-data/state_speech/tmp_models/beta.npy --units "$PWD"/ocr-data/state_speech/tmp_models/beta_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/beta_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/beta_timestamp.npy
    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_cauchy.py --cauchy "$PWD"/ocr-data/state_speech/tmp_models/cauchy.npy --units "$PWD"/ocr-data/state_speech/tmp_models/cauchy_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/cauchy_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/cauchy_timestamp.npy
    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_gamma.py --gamma "$PWD"/ocr-data/state_speech/tmp_models/gamma.npy --units "$PWD"/ocr-data/state_speech/tmp_models/gamma_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/gamma_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/gamma_timestamp.npy
    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_rayleigh.py --rayleigh "$PWD"/ocr-data/state_speech/tmp_models/rayleigh.npy --units "$PWD"/ocr-data/state_speech/tmp_models/rayleigh_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/rayleigh_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/rayleigh_timestamp.npy
    python2.7 "$PWD"/qsr_lib_ocr/qsr_lib/scripts/qstag_example_weibull.py --weibull "$PWD"/ocr-data/state_speech/tmp_models/weibull.npy --units "$PWD"/ocr-data/state_speech/tmp_models/weibull_units.npy --scores "$PWD"/ocr-data/state_speech/tmp_models/weibull_scores.npy --timestamp "$PWD"/ocr-data/state_speech/tmp_models/weibull_timestamp.npy
else
    echo "No distribution specified.\n"
    exit
fi