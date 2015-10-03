
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>

#include <FactorGraphModel.h>
#include <FactorGraph.h>
#include <FactorType.h>
#include <NormalPrior.h>
#include <LaplacePrior.h>
#include <StudentTPrior.h>
#include <MaximumLikelihood.h>
#include <MaximumPseudolikelihood.h>
#include <NonlinearRBFFactorType.h>
#include <TreeInference.h>
#include <GibbsInference.h>

int main(int argc, char* argv[]) {
	Grante::FactorGraphModel model;

	// Create one unary factor type: letter 'a'-'z'
	std::vector<unsigned int> card(1, 26);
	std::vector<double> w(26*16*8, 0.0);
	bool use_rbf = false;

	Grante::FactorType* ft = 0;
	if (use_rbf) {
		unsigned int rbf_basis_count = 16;
		ft = new Grante::NonlinearRBFFactorType("letter_unary", card,
			/*data_size*/ 16*8, rbf_basis_count,
			/*log_beta*/ std::log(1.0 / 64.0));
	} else {
		ft = new Grante::FactorType("letter_unary", card, w);
	}
	model.AddFactorType(ft);

	// Create the pairwise factor
	std::vector<unsigned int> card_pw(2, 26);
	std::vector<double> w_pw(26*26, 0.0);
	Grante::FactorType* ft_pw =
		new Grante::FactorType("pairwise", card_pw, w_pw);
	model.AddFactorType(ft_pw);

	// Get the factor type back from the model
	Grante::FactorType* pt = model.FindFactorType("letter_unary");
	Grante::FactorType* pt_pw = model.FindFactorType("pairwise");

	// Read in training data
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		test_data;
	std::vector<Grante::InferenceMethod*> inference_methods;

	std::ifstream dfs;
	dfs.open("letter.data");
	if (dfs.good() == false) {
		std::cerr << "Failed to open 'letter.data' data file." << std::endl;
		return (1);
	}

	std::cout << "Reading data..." << std::endl;
	std::string word;

	// fg_folds[cv_fold] = set of factor graphs in this fold
	std::map<unsigned int, std::vector<
		Grante::ParameterEstimationMethod::labeled_instance_type> > fg_folds;
	std::map<unsigned int, std::vector<Grante::InferenceMethod*> > fg_inf;

	std::vector<std::vector<double> > data;
	while (dfs.eof() == false) {
		unsigned int id;	// unique id of this letter
		dfs >> id;
		if (dfs.eof())
			break;

		unsigned char letter_c;	// 'a' to 'z'
		dfs >> letter_c;
		assert(letter_c >= 'a' && letter_c <= 'z');
		word += letter_c;

		int next_id;	// -1 if end of word
		dfs >> next_id;

		int word_id;	// word id (starting with 1)
		dfs >> word_id;

		int word_pos;	// position of letter in word (starting with 1)
		dfs >> word_pos;

		unsigned int fold_id;	// cross-validation fold id of this word
		dfs >> fold_id;

		std::vector<double> data_cur(16*8, 0.0);
		for (unsigned int n = 0; n < 16*8; ++n)
			dfs >> data_cur[n];
		data.push_back(data_cur);

		if (next_id != -1)
			continue;

		// Add sample
		std::vector<unsigned int> fg_varcard(word.length(), 26);
		Grante::FactorGraph* fg = new Grante::FactorGraph(&model, fg_varcard);

		// Add unary observation factors
		for (unsigned int ci = 0; ci < word.length(); ++ci) {
			std::vector<unsigned int> factor_varindex(1, ci);
			Grante::Factor* fac =
				new Grante::Factor(pt, factor_varindex, data[ci]);
			fg->AddFactor(fac);

			if (ci == 0)
				continue;

			// Add pairwise factor between (ci-1) and (ci)
			std::vector<unsigned int> fac_pw_varindex(2);
			fac_pw_varindex[0] = ci-1;
			fac_pw_varindex[1] = ci;
			std::vector<double> data_pw_empty;
			Grante::Factor* fac_pw =
				new Grante::Factor(pt_pw, fac_pw_varindex, data_pw_empty);
			fg->AddFactor(fac_pw);
		}

		// Compute ground truth label
		std::vector<unsigned int> fg_label(word.length(), 0);
		for (unsigned int wi = 0; wi < word.length(); ++wi)
			fg_label[wi] = word[wi] - 'a';	// [0-25]

		// Add factor graph and inference method for chain
		Grante::ParameterEstimationMethod::labeled_instance_type lit(fg,
			new Grante::FactorGraphObservation(fg_label));
		fg_folds[fold_id].push_back(lit);
		fg_inf[fold_id].push_back(new Grante::TreeInference(fg));

		// Full word
		word.clear();
		data.clear();
	}

	dfs.close();

	// Train using maximum likelihood
	std::cout << "Cross validation performance estimation..." << std::endl;
	double all_acc = 0.0;
	double all_count = 0.0;
	double all_acc_squared = 0.0;
	for (unsigned int fold = 0; fold < 10; ++fold) {
		std::cout << "   Fold " << (fold+1) << " of 10: " << std::endl;

		// Produce training data
		std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
			training_data;
		std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
			test_data;
		std::vector<Grante::InferenceMethod*> inference_methods;
		for (unsigned int fi = 0; fi < 10; ++fi) {
			if (fi != fold) {
				// Test data
				test_data.insert(test_data.end(),
					fg_folds[fi].begin(), fg_folds[fi].end());
			} else {
				// Training data
				training_data.reserve(training_data.size() +
					fg_folds[fi].size());
				training_data.insert(training_data.end(),
					fg_folds[fi].begin(), fg_folds[fi].end());
				inference_methods.reserve(inference_methods.size() +
					fg_inf[fi].size());
				inference_methods.insert(inference_methods.end(),
					fg_inf[fi].begin(), fg_inf[fi].end());
			}
		}
		std::cout << training_data.size() << " training instances, "
			<< test_data.size() << " test instances." << std::endl;

		// Initialize RBF network
		if (use_rbf) {
			dynamic_cast<Grante::NonlinearRBFFactorType*>(
				ft)->InitializeUsingTrainingData(training_data);
		}

		//Grante::MaximumPseudolikelihood mle(&model);
		Grante::MaximumLikelihood mle(&model);
		mle.SetupTrainingData(training_data, inference_methods);
//		mle.AddPrior("letter_unary", new Grante::NormalPrior(1.0, w.size()));
		mle.AddPrior("pairwise", new Grante::NormalPrior(1.0, w_pw.size()));
		mle.Train(1e-4);
		std::cout << "Finished training." << std::endl;

		// Perform inference on current test sample
		unsigned int tc_count = 0;
		unsigned int tc_correct = 0;
		for (unsigned int n = 0; n < test_data.size(); ++n) {
			// Perform MAP prediction
			//Grante::TreeInference tinf(test_data[n].first);
			Grante::GibbsInference tinf(test_data[n].first);
std::cout<<"GibbsInference method"<<std::endl;			
			test_data[n].first->ForwardMap();
			std::vector<unsigned int> map_state;
			tinf.MinimizeEnergy(map_state);
			const std::vector<unsigned int>& true_state =
				test_data[n].second->State();
			std::string word_correct;
			std::string word_pred;
			for (unsigned int ci = 0; ci < map_state.size(); ++ci) {
				word_correct += ('a' + static_cast<char>(true_state[ci]));
				word_pred += ('a' + static_cast<char>(map_state[ci]));
				if (true_state[ci] == map_state[ci])
					tc_correct += 1;
				tc_count += 1;
			}
			std::cout << "TRUE: '" << word_correct << "', PRED: '"
				<< word_pred << "'" << std::endl;
		}
		double test_acc = 100.0 * (static_cast<double>(tc_correct) /
			static_cast<double>(tc_count));
		std::cout << "Test per-char accuracy: " << test_acc
			<< " percent." << std::endl;
		all_acc += test_acc;
		all_count += 1.0;
		all_acc_squared += test_acc*test_acc;
	}
	double mean_acc = all_acc / all_count;
	// standard deviation of the cv-sample
	double sdev_sample = std::sqrt(all_acc_squared/10.0 -
		mean_acc*mean_acc);
	std::cout << "Mean test accuracy: " << mean_acc << " +/- "
		<< sdev_sample << " percent."
		<< std::endl;
}

