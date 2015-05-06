// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/syncedmem.hpp"
#include <ctime>
#include <cstdio>
#include <string.h>
#include <locale>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
	void APLLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		CHECK_GE(bottom[0]->num_axes(), 2)
			<< "Number of axes of bottom blob must be >=2.";

		// Figure out the dimensions
		M_ = bottom[0]->num();
		K_ = bottom[0]->count() / bottom[0]->num();
		N_ = K_;

		sums_ = this->layer_param_.apl_param().sums();
		save_mem_ = this->layer_param_.apl_param().save_mem();

		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		} 

		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		} else {
			this->blobs_.resize(2);

			shared_ptr<Filler<Dtype> > slope_filler;
			if (this->layer_param_.apl_param().has_slope_filler()) {
				slope_filler.reset(GetFiller<Dtype>(this->layer_param_.apl_param().slope_filler()));
			} else {
				FillerParameter slope_filler_param;
				slope_filler_param.set_type("uniform");
				slope_filler_param.set_min((Dtype) -0.5/((Dtype) sums_));
				slope_filler_param.set_max((Dtype)  0.5/((Dtype) sums_));
				slope_filler.reset(GetFiller<Dtype>(slope_filler_param));
			}
			//shared_ptr<Filler<Dtype> > slope_filler(GetFiller<Dtype>(
			//		this->layer_param_.apl_param().slope_filler()));
			shared_ptr<Filler<Dtype> > offset_filler;
			if (this->layer_param_.apl_param().has_offset_filler()) {
				offset_filler.reset(GetFiller<Dtype>(this->layer_param_.apl_param().offset_filler()));
			} else {
				FillerParameter offset_filler_param;
				offset_filler_param.set_type("gaussian");
				offset_filler_param.set_std(0.5);
				offset_filler.reset(GetFiller<Dtype>(offset_filler_param));
			}
			//shared_ptr<Filler<Dtype> > offset_filler(GetFiller<Dtype>(
			//		this->layer_param_.apl_param().offset_filler()));

			//slope
			this->blobs_[0].reset(new Blob<Dtype>(1, 1, sums_, K_));
			CHECK(this->blobs_[0].get()->count());
			slope_filler->Fill(this->blobs_[0].get());

			//offset
			this->blobs_[1].reset(new Blob<Dtype>(1, 1, sums_, K_));
			CHECK(this->blobs_[1].get()->count());
			offset_filler->Fill(this->blobs_[1].get());
		}


		if (!save_mem_) {
			temp_ex_neuron_sum_.reset(new SyncedMemory(M_ * K_ * sums_ * sizeof(Dtype)));

			example_multiplier_.reset(new SyncedMemory(M_ * sizeof(Dtype)));
			Dtype* example_multiplier_data =
				reinterpret_cast<Dtype*>(example_multiplier_->mutable_cpu_data());
			for (int i = 0; i < M_; ++i) {
				example_multiplier_data[i] = 1.;
			}
		}

		maxs_.reset(new SyncedMemory(M_ * K_ * sums_ * sizeof(Dtype)));

		LOG(INFO) << " Sums: " << sums_;
	}

template <typename Dtype>
	void APLLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		CHECK_GE(bottom[0]->num_axes(), 2)
			<< "Number of axes of bottom blob must be >=2.";
		top[0]->ReshapeLike(*bottom[0]);

		if (bottom[0] == top[0]) {
			// For in-place computation
			inPlace_memory_.ReshapeLike(*bottom[0]);
		}
	}

template <typename Dtype>
	void APLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		/* Initialize */
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		const Dtype* neuron_weight = this->blobs_[0]->cpu_data();
		const Dtype* neuron_offset = this->blobs_[1]->cpu_data();
		const int count = bottom[0]->count();

		Dtype* maxs_data = reinterpret_cast<Dtype*>(maxs_->mutable_cpu_data());

		// For in-place computation
		if (bottom[0] == top[0]) {
			caffe_copy(count, bottom_data, inPlace_memory_.mutable_cpu_data());
		}

		/* Forward Prop */
		for (int e=0; e<M_; ++e) {
			int exPos = e*K_;
			int exPosSums = e*K_*sums_;
			for (int k=0; k<K_; ++k) {
				Dtype bottom_data_ex = bottom_data[exPos + k];
				top_data[exPos + k] = max(bottom_data_ex,Dtype(0));

				int sumPos = k*sums_;
				for (int s=0; s<sums_; ++s) {
					maxs_data[exPosSums + sumPos + s] = max(-bottom_data_ex + neuron_offset[sumPos + s], Dtype(0));
					top_data[exPos + k] += neuron_weight[sumPos + s]*maxs_data[exPosSums + sumPos + s];
				}
			}
		}
	}

template <typename Dtype>
	void APLLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom) {
		/* Initialize */
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const Dtype* neuron_weight = this->blobs_[0]->cpu_data();

		Dtype* neuron_weight_diff = this->blobs_[0]->mutable_cpu_diff();
		Dtype* neuron_offset_diff = this->blobs_[1]->mutable_cpu_diff();

		const Dtype* maxs_data = reinterpret_cast<const Dtype*>(maxs_->cpu_data());

		// For in-place computation
		if (top[0] == bottom[0]) {
			bottom_data = inPlace_memory_.cpu_data();
		}

		for (int i=0; i < sums_*K_; ++i) {
			neuron_weight_diff[i] = 0;
			neuron_offset_diff[i] = 0;
		}

		/* Gradients to neuron layer*/
		for (int e=0; e<M_; ++e) {
			int exPos = e*K_;
			int exPosSums = e*K_*sums_;

			for (int k=0; k<K_; ++k) {
				Dtype sumTopDiff = top_diff[exPos + k];
				Dtype sumBottomData = bottom_data[exPos + k];			

				//bottom_diff[exPos + k] = sumTopDiff*(sumBottomData > 0);
				bottom_diff[exPos + k] = sumBottomData > 0 ? sumTopDiff : 0;

				int sumPos = k*sums_;
				for (int s=0; s<sums_; ++s) {
					Dtype maxGT_Zero = maxs_data[exPosSums + sumPos + s] > 0;

					Dtype weight_diff = sumTopDiff*maxs_data[exPosSums + sumPos + s];
					Dtype offset_diff = sumTopDiff*neuron_weight[sumPos + s]*maxGT_Zero;

					neuron_weight_diff[sumPos + s] += weight_diff;
					neuron_offset_diff[sumPos + s] += offset_diff;

					//Propagate down gradients to lower layer
					bottom_diff[exPos + k] += -offset_diff;
				}
			}
		}
	}

#ifdef CPU_ONLY
STUB_GPU(APLLayer);
#endif

INSTANTIATE_CLASS(APLLayer);
REGISTER_LAYER_CLASS(APL);

}  // namespace caffe
