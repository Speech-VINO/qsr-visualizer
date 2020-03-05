path27=$(which python2.7)
path3=$(which python3)

if test -z "$path27"
then
      echo "\$python2.7 is missing. Please install python 2.7"
      return
else
      echo "python2.7 available.\n"
      python2.7 -m pip install --user "ruamel.yaml"
      python2.7 -m pip install --user matplotlib seaborn
fi

if test -z "$path3"
then
      echo "\npython3 is missing. Please install python3\n"
      return
else
      echo "python3 available.\n"
      echo "Installing python3 packages..\n"
      python3 -m pip install --user "tensorflow>=2.0.0"
fi

while getopts "d:" opt; do
  case $opt in
    d)  
        dist=$OPTARG;;
    *) 
        echo "Invalid option: -$OPTARG" >&2;;  
  esac
done

if test -z "$dist"
then
    echo "No distribution specified"
    return
fi

echo "Analysing the state space.. \nIt trains a Tensorflow model and produces plots.."

python3 speech/state_space/execute.py --distribution "$dist"
python3 speech/state_space/plot.py --distribution "$dist"

if [ "$dist" == "beta" ]
then

    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_beta.py --plot_episodes --beta "$PWD"/speech/state_space/tmp_models/beta.npy --units "$PWD"/speech/state_space/tmp_models/beta_units.npy --scores "$PWD"/speech/state_space/tmp_models/beta_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/beta_timestamp.npy tpcc

elif [ "$dist" == "cauchy" ]
then

    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_cauchy.py --plot_episodes --cauchy "$PWD"/speech/state_space/tmp_models/cauchy.npy --units "$PWD"/speech/state_space/tmp_models/cauchy_units.npy --scores "$PWD"/speech/state_space/tmp_models/cauchy_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/cauchy_timestamp.npy tpcc

elif [ "$dist" == "gamma" ]
then

    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_gamma.py --plot_episodes --gamma "$PWD"/speech/state_space/tmp_models/gamma.npy --units "$PWD"/speech/state_space/tmp_models/gamma_units.npy --scores "$PWD"/speech/state_space/tmp_models/gamma_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/gamma_timestamp.npy tpcc

elif [ "$dist" == "rayleigh" ]
then

    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_rayleigh.py --plot_episodes --rayleigh "$PWD"/speech/state_space/tmp_models/rayleigh.npy --units "$PWD"/speech/state_space/tmp_models/rayleigh_units.npy --scores "$PWD"/speech/state_space/tmp_models/rayleigh_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/rayleigh_timestamp.npy tpcc

elif [ "$dist" == "weibull" ]
then

    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_weibull.py --plot_episodes --weibull "$PWD"/speech/state_space/tmp_models/weibull.npy --units "$PWD"/speech/state_space/tmp_models/weibull_units.npy --scores "$PWD"/speech/state_space/tmp_models/weibull_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/weibull_timestamp.npy tpcc

elif [ "$dist" == "all" ]
then

    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_beta.py --plot_episodes --beta "$PWD"/speech/state_space/tmp_models/beta.npy --units "$PWD"/speech/state_space/tmp_models/beta_units.npy --scores "$PWD"/speech/state_space/tmp_models/beta_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/beta_timestamp.npy tpcc
    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_cauchy.py --plot_episodes --cauchy "$PWD"/speech/state_space/tmp_models/cauchy.npy --units "$PWD"/speech/state_space/tmp_models/cauchy_units.npy --scores "$PWD"/speech/state_space/tmp_models/cauchy_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/cauchy_timestamp.npy tpcc
    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_gamma.py --plot_episodes --gamma "$PWD"/speech/state_space/tmp_models/gamma.npy --units "$PWD"/speech/state_space/tmp_models/gamma_units.npy --scores "$PWD"/speech/state_space/tmp_models/gamma_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/gamma_timestamp.npy tpcc
    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_rayleigh.py --plot_episodes --rayleigh "$PWD"/speech/state_space/tmp_models/rayleigh.npy --units "$PWD"/speech/state_space/tmp_models/rayleigh_units.npy --scores "$PWD"/speech/state_space/tmp_models/rayleigh_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/rayleigh_timestamp.npy tpcc
    python2.7 "$PWD"/qsr_lib_speech/qsr_lib/scripts/qstag_example_weibull.py --plot_episodes --weibull "$PWD"/speech/state_space/tmp_models/weibull.npy --units "$PWD"/speech/state_space/tmp_models/weibull_units.npy --scores "$PWD"/speech/state_space/tmp_models/weibull_scores.npy --timestamp "$PWD"/speech/state_space/tmp_models/weibull_timestamp.npy tpcc

else
    echo "No distribution specified.\n"
    return
fi