// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! WRITEME

use crate::{array::PrimitiveArray, datatypes::ArrowPrimitiveType};

/// WRITEME
#[derive(Debug)]
pub enum Datum<'d, T>
where
    T: ArrowPrimitiveType,
{
    Array(&'d PrimitiveArray<T>),
    Scalar(Option<T::Native>),
}

impl<'d, T> From<&'d PrimitiveArray<T>> for Datum<'d, T>
where
    T: ArrowPrimitiveType,
{
    fn from(array: &'d PrimitiveArray<T>) -> Self {
        Datum::Array(array)
    }
}

impl<'d, T> From<Option<T::Native>> for Datum<'d, T>
where
    T: ArrowPrimitiveType,
{
    fn from(scalar: Option<T::Native>) -> Self {
        Datum::Scalar(scalar)
    }
}
