#include "Model.h"

float Binary_Cross_Entropy_Cost(const float* hx, const float* _Y, const int n)
{
    //std::cout << "Binary_CEC\n";
    Eigen::Map<VectorXf> hx_vec(const_cast<float*>(hx), n);
    Eigen::Map<VectorXf> y_vec(const_cast<float*>(_Y), n);
    float cost = (y_vec.array() * log(hx_vec.array()) + (1 - y_vec.array()) * log(1 - hx_vec.array())).sum();
    cost = (-1.0 / n) * cost;

    return cost;
}

float Mean_Squared_Error(const float* hx, const float* _Y, const int n)
{
    //std::cout << "MSE\n";
    Eigen::Map<VectorXf> hx_vec(const_cast<float*>(hx), n); 
    Eigen::Map<VectorXf> y_vec(const_cast<float*>(_Y), n);
    float cost = (y_vec - hx_vec).cwiseAbs2().sum();
    cost = (1.0 / n) * cost;

    return cost;
}

float Mean_Absolute_Error(const float* hx, const float* _Y, const int n)
{
    //std::cout << "MAE\n";
    Eigen::Map<VectorXf> hx_vec(const_cast<float*>(hx), n);
    Eigen::Map<VectorXf> y_vec(const_cast<float*>(_Y), n);
    float cost = (y_vec - hx_vec).cwiseAbs().sum();
    cost = (1.0 / n) * cost;

    return cost;
}

float Categorical_Cross_Entropy_Cost(const float* hx, const float* _Y, const int n)
{
    //std::cout << "Categorical_CEC\n";
    return 0.0f;
}



bool Model::Initialize()
{
    int n = model_layers_param.size();
    Univ_Layer_Param tmp;
    Eigen::VectorXi next_inp_shp = m_in_shape;

    for (int i = 0; i < n; i++) {
        switch (model_layers_param[i].l_type) {
        case LAYER_TYPE::DENSE: {
            tmp = model_layers_param[i];
            model_layers.push_back(new Dense_FC(next_inp_shp.prod(), tmp.params[0],
                static_cast<ACT_TYPE>(tmp.params[1]), static_cast<REGULARIZATION>(tmp.params[2])));
            next_inp_shp = model_layers[i]->Out_shape(next_inp_shp);
            break;
        }
        case LAYER_TYPE::MAX_POOL: {
            if (i > 0)
                if (model_layers_param[i - 1].l_type == LAYER_TYPE::DENSE)return false;
            if (next_inp_shp.size() != 3) return false;

            tmp = model_layers_param[i];
            model_layers.push_back(new Max_Pool_2D(tmp.params[0], tmp.params[1], tmp.params[2]));
            next_inp_shp = model_layers[i]->Out_shape(next_inp_shp);
            break;
        }
        case LAYER_TYPE::CONV: {
            if (i > 0)
                if (model_layers_param[i - 1].l_type == LAYER_TYPE::DENSE)return false;
            if (next_inp_shp.size() != 3) return false;

            tmp = model_layers_param[i];
            model_layers.push_back(new Convolution(tmp.params[0], tmp.params[1], next_inp_shp(0), tmp.params[2],
                static_cast<ACT_TYPE>(tmp.params[3]), static_cast<REGULARIZATION>(tmp.params[4])));
            next_inp_shp = model_layers[i]->Out_shape(next_inp_shp);
            break;
        }
        default: {return false; }
        }
    }

    model_layers.shrink_to_fit();
    m_init = true;
    return true;
}

float Model::model_Forward_Pass(Tensor4f& _X, const Tensor4f& _Y)
{// Just for Testing ///

    if (!m_init)return -1;

    int n_layers = model_layers.size();
    Tensor4f* _A = new Tensor4f[n_layers];

    if (!model_layers[0]->Forward_Pass(_X, _A[0])) {
        std::cout << _A[0].dimensions() << '\n';
        delete[] _A;
        return -1;
    }
    for (int i = 1; i < n_layers; i++) {
        if (!model_layers[i]->Forward_Pass(_A[i-1], _A[i])) {
            std::cout << _A[i].dimensions() << '\n';
            delete[] _A;
            return -1;
        }    
    }

    float cost = cost_Function(_A[n_layers - 1].data(), _Y.data(), _A[n_layers - 1].size());

    delete[] _A;
    return cost;
}

bool Model::Run(Tensor4f& _X, const Tensor4f& _Y, int batch_size, int epochs)
{
    if (!m_init)return false;

    int n_layers = model_layers.size();
    Tensor4f* _A = new Tensor4f[n_layers];

    /*if (!model_layers[0]->Forward_Pass(_X, _A[0])) {
        std::cout << _A[0].dimensions() << '\n';
        delete[] _A;
        return false;
    }
    for (int i = 1; i < n_layers; i++) {
        if (!model_layers[i]->Forward_Pass(_A[i - 1], _A[i])) {
            std::cout << _A[i].dimensions() << '\n';
            delete[] _A;
            return false;
        }
    }*/

    const int m = _A[n_layers-1].dimension(0);
    Tensor4f dA_prev;// = -((_Y / _A[n_layers - 1]) - (1.0 - _Y) / (1.0 - _A[n_layers - 1])); // Need to automate 
    Tensor4f dX;

   /* auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto time_taken_1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);*/
    

    // Optimize //
    //for (int i = n_layers - 1; i >= 0; i--) {

    //    //start = std::chrono::high_resolution_clock::now();
    //    bool res = model_layers[i]->Backward_Pass(dA_prev, dX);
    //    //dA_prev = dX;
    //   // stop = std::chrono::high_resolution_clock::now();
    //    //time_taken_1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    //    if (!res)
    //    {
    //        std::cout << "Error in Backprop\n";
    //        model_layers[i]->Summary();
    //        delete[] _A;
    //        return false;
    //    }
    //    /*else
    //        std::cout << dA_prev.dimensions() << "  " << dX.dimensions() << "  Time:  " << time_taken_1.count() << '\n';*/

    //    dA_prev = dX;
    //}
    
    for (int n = 0; n < epochs; n++) {
        // Forward Pass //
        if (!model_layers[0]->Forward_Pass(_X, _A[0])) {
            std::cout << _A[0].dimensions() << '\n';
            delete[] _A;
            return false;
        }
        for (int i = 1; i < n_layers; i++) {
            if (!model_layers[i]->Forward_Pass(_A[i - 1], _A[i])) {
                std::cout << _A[i].dimensions() << '\n';
                delete[] _A;
                return false;
            }
        }

        float cost = cost_Function(_A[n_layers - 1].data(), _Y.data(), _A[n_layers - 1].size());

        dA_prev = -((_Y / _A[n_layers - 1]) - (1.0 - _Y) / (1.0 - _A[n_layers - 1]));

        // Backward Pass //
        for (int i = n_layers - 1; i >= 0; i--) {
            bool res = model_layers[i]->Backward_Pass(dA_prev, dX);

            if (!res)
            {
                std::cout << "Error in Backprop\n";
                model_layers[i]->Summary();
                delete[] _A;
                return false;
            }

            dA_prev = dX;
        }

        // Weight Update //
        for (int i = 0; i < n_layers; i++)
            model_layers[i]->Update_Weights(0.01, REGULARIZATION::L2);

        std::cout << cost << " Epoch: " << n + 1 << '\n';

    }

    delete[] _A;
    return true;
}

void Model::Model_Summary()
{
    if (!m_init) {
        std::cout << "Model Not Initialized\n";
        return;
    }

    for (const auto& el : model_layers)
        el->Summary();

}
