/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bindings.h"
#include <nanobind/stl/string.h>

#include "tensorrt_llm/common/bindingUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::process_group
{

void initBindings(nb::module_& m)
{

    m.def("init_pg",
        [](nb::object world_pg_obj, nb::object local_pg_obj, std::string const& pybind11_abi)
        {
            using Pg = c10d::ProcessGroup;
            using E = nb::python_error;

            pg_utils::init_pg(common::get_intrusive_ptr<Pg, E>(world_pg_obj.ptr(), pybind11_abi),
                common::get_intrusive_ptr<Pg, E>(local_pg_obj.ptr(), pybind11_abi));
        });

    nb::class_<pg_utils::LocalNodeBarrier>(m, "LocalNodeBarrier")
        .def("__init__",
            [](pg_utils::LocalNodeBarrier* self, std::string mem_path)
            {
                new (self) pg_utils::LocalNodeBarrier(std::move(mem_path));
                self->init();
            })
        .def("spin_sync", &pg_utils::LocalNodeBarrier::spin_sync, nb::call_guard<nb::gil_scoped_release>())
        .def("futex_sync", &pg_utils::LocalNodeBarrier::futex_sync, nb::call_guard<nb::gil_scoped_release>());
}

} // namespace tensorrt_llm::nanobind::process_group
