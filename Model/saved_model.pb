??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
8
Const
output"dtype"
valuetensor"
dtypetype
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
n
identifiersVarHandleOp*
_output_shapes
: *
dtype0*
shape:G*
shared_nameidentifiers
g
identifiers/Read/ReadVariableOpReadVariableOpidentifiers*
_output_shapes
:G*
dtype0
p

candidatesVarHandleOp*
_output_shapes
: *
dtype0*
shape
:G *
shared_name
candidates
i
candidates/Read/ReadVariableOpReadVariableOp
candidates*
_output_shapes

:G *
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameembedding/embeddings
}
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes

: *
dtype0
j

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name71*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_1Const*
_output_shapes
:*
dtype0*[
valueRBPB1B11B15B2B3B36B39B4B44B46B47B48B49B50B53B54B55B56B57
?
Const_2Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                        	       
                                                                      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_1Const_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *"
fR
__inference_<lambda>_2145
&
NoOpNoOp^StatefulPartitionedCall
?
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
query_model
identifiers
_identifiers

candidates
_candidates
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
query_with_exclusions

signatures*
?
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
KE
VARIABLE_VALUEidentifiers&identifiers/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUE
candidates%candidates/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0*
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 
* 

serving_default* 
#
lookup_table
	keras_api* 
?

embeddings
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*

0*

0*
* 
?
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEembedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0*
* 
* 
* 
* 
R
)_initializer
*_create_resource
+_initialize
,_destroy_resource* 
* 

0*

0*
* 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
r
serving_default_input_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_1
hash_tableConstembedding/embeddings
candidatesidentifiers*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_2055
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameidentifiers/Read/ReadVariableOpcandidates/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOpConst_3*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_2183
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameidentifiers
candidatesembedding/embeddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_2202??
?
?
)__inference_sequential_layer_call_fn_1780
string_lookup_input
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1760o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1956
input_1
sequential_1938
sequential_1940	!
sequential_1942: 0
matmul_readvariableop_resource:G 
gather_resource:G

identity_1

identity_2??Gather?MatMul/ReadVariableOp?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1938sequential_1940sequential_1942*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1760t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:G *
dtype0?
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_2066

inputs
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_2077

inputs
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1760o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
|
(__inference_embedding_layer_call_fn_2110

inputs	
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_1714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
"__inference_signature_wrapper_2055
input_1
unknown
	unknown_0	
	unknown_1: 
	unknown_2:G 
	unknown_3:G
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_1694o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1760

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_1756: 
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1756*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_1714y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
__inference_<lambda>_21455
1key_value_init70_lookuptableimportv2_table_handle-
)key_value_init70_lookuptableimportv2_keys/
+key_value_init70_lookuptableimportv2_values	
identity??$key_value_init70/LookupTableImportV2?
$key_value_init70/LookupTableImportV2LookupTableImportV21key_value_init70_lookuptableimportv2_table_handle)key_value_init70_lookuptableimportv2_keys+key_value_init70_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init70/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$key_value_init70/LookupTableImportV2$key_value_init70/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
 __inference__traced_restore_2202
file_prefix*
assignvariableop_identifiers:G/
assignvariableop_1_candidates:G 9
'assignvariableop_2_embedding_embeddings: 

identity_4??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHx
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_identifiersIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_candidatesIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp'assignvariableop_2_embedding_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_4IdentityIdentity_3:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_4Identity_4:output:0*
_input_shapes

: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1802
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_1798: 
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1798*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_1714y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1882
queries
sequential_1864
sequential_1866	!
sequential_1868: 0
matmul_readvariableop_resource:G 
gather_resource:G

identity_1

identity_2??Gather?MatMul/ReadVariableOp?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_1864sequential_1866sequential_1868*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1760t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:G *
dtype0?
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?	
?
*__inference_brute_force_layer_call_fn_1914
input_1
unknown
	unknown_0	
	unknown_1: 
	unknown_2:G 
	unknown_3:G
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_1882o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1719

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_1715: 
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1715*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_1714y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
C__inference_embedding_layer_call_and_return_conditional_losses_2119

inputs	'
embedding_lookup_2113: 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2113inputs*
Tindices0	*(
_class
loc:@embedding_lookup/2113*'
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/2113*'
_output_shapes
:????????? }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_brute_force_layer_call_fn_1990
queries
unknown
	unknown_0	
	unknown_1: 
	unknown_2:G 
	unknown_3:G
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_1882o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_2013
queriesG
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	<
*sequential_embedding_embedding_lookup_1997: 0
matmul_readvariableop_resource:G 
gather_resource:G

identity_1

identity_2??Gather?MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?6sequential/string_lookup/None_Lookup/LookupTableFindV2?
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handlequeriesDsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_1997*sequential/string_lookup/Identity:output:0*
Tindices0	*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1997*'
_output_shapes
:????????? *
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/1997*'
_output_shapes
:????????? ?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:G *
dtype0?
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1827
queries
sequential_1809
sequential_1811	!
sequential_1813: 0
matmul_readvariableop_resource:G 
gather_resource:G

identity_1

identity_2??Gather?MatMul/ReadVariableOp?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallqueriessequential_1809sequential_1811sequential_1813*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1719t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:G *
dtype0?
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1935
input_1
sequential_1917
sequential_1919	!
sequential_1921: 0
matmul_readvariableop_resource:G 
gather_resource:G

identity_1

identity_2??Gather?MatMul/ReadVariableOp?"sequential/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1917sequential_1919sequential_1921*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1719t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:G *
dtype0?
MatMulMatMul+sequential/StatefulPartitionedCall:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^Gather^MatMul/ReadVariableOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
9
__inference__creator_2124
identity??
hash_tablej

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name71*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference__wrapped_model_1694
input_1S
Obrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_table_handleT
Pbrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_default_value	H
6brute_force_sequential_embedding_embedding_lookup_1678: <
*brute_force_matmul_readvariableop_resource:G )
brute_force_gather_resource:G
identity

identity_1??brute_force/Gather?!brute_force/MatMul/ReadVariableOp?1brute_force/sequential/embedding/embedding_lookup?Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2?
Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Obrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_table_handleinput_1Pbrute_force_sequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
-brute_force/sequential/string_lookup/IdentityIdentityKbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
1brute_force/sequential/embedding/embedding_lookupResourceGather6brute_force_sequential_embedding_embedding_lookup_16786brute_force/sequential/string_lookup/Identity:output:0*
Tindices0	*I
_class?
=;loc:@brute_force/sequential/embedding/embedding_lookup/1678*'
_output_shapes
:????????? *
dtype0?
:brute_force/sequential/embedding/embedding_lookup/IdentityIdentity:brute_force/sequential/embedding/embedding_lookup:output:0*
T0*I
_class?
=;loc:@brute_force/sequential/embedding/embedding_lookup/1678*'
_output_shapes
:????????? ?
<brute_force/sequential/embedding/embedding_lookup/Identity_1IdentityCbrute_force/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? ?
!brute_force/MatMul/ReadVariableOpReadVariableOp*brute_force_matmul_readvariableop_resource*
_output_shapes

:G *
dtype0?
brute_force/MatMulMatMulEbrute_force/sequential/embedding/embedding_lookup/Identity_1:output:0)brute_force/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G*
transpose_b(V
brute_force/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
?
brute_force/TopKV2TopKV2brute_force/MatMul:product:0brute_force/TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
?
brute_force/GatherResourceGatherbrute_force_gather_resourcebrute_force/TopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype0o
brute_force/IdentityIdentitybrute_force/Gather:output:0*
T0*'
_output_shapes
:?????????
j
IdentityIdentitybrute_force/TopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
n

Identity_1Identitybrute_force/Identity:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^brute_force/Gather"^brute_force/MatMul/ReadVariableOp2^brute_force/sequential/embedding/embedding_lookupC^brute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2(
brute_force/Gatherbrute_force/Gather2F
!brute_force/MatMul/ReadVariableOp!brute_force/MatMul/ReadVariableOp2f
1brute_force/sequential/embedding/embedding_lookup1brute_force/sequential/embedding/embedding_lookup2?
Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2Bbrute_force/sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1791
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	 
embedding_1787: 
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1787*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_1714y
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
__inference__traced_save_2183
file_prefix*
&savev2_identifiers_read_readvariableop)
%savev2_candidates_read_readvariableop3
/savev2_embedding_embeddings_read_readvariableop
savev2_const_3

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&identifiers/.ATTRIBUTES/VARIABLE_VALUEB%candidates/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHu
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_identifiers_read_readvariableop%savev2_candidates_read_readvariableop/savev2_embedding_embeddings_read_readvariableopsavev2_const_3"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*1
_input_shapes 
: :G:G : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:G:$ 

_output_shapes

:G :$ 

_output_shapes

: :

_output_shapes
: 
?
?
E__inference_brute_force_layer_call_and_return_conditional_losses_2036
queriesG
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	<
*sequential_embedding_embedding_lookup_2020: 0
matmul_readvariableop_resource:G 
gather_resource:G

identity_1

identity_2??Gather?MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?6sequential/string_lookup/None_Lookup/LookupTableFindV2?
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handlequeriesDsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_2020*sequential/string_lookup/Identity:output:0*
Tindices0	*=
_class3
1/loc:@sequential/embedding/embedding_lookup/2020*'
_output_shapes
:????????? *
dtype0?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/2020*'
_output_shapes
:????????? ?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:G *
dtype0?
MatMulMatMul9sequential/embedding/embedding_lookup/Identity_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????G*
transpose_b(J
TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :
z
TopKV2TopKV2MatMul:product:0TopKV2/k:output:0*
T0*:
_output_shapes(
&:?????????
:?????????
?
GatherResourceGathergather_resourceTopKV2:indices:0*
Tindices0*'
_output_shapes
:?????????
*
dtype0W
IdentityIdentityGather:output:0*
T0*'
_output_shapes
:?????????
`

Identity_1IdentityTopKV2:values:0^NoOp*
T0*'
_output_shapes
:?????????
b

Identity_2IdentityIdentity:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^Gather^MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 2
GatherGather2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV2:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
+
__inference__destroyer_2137
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_2103

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	1
embedding_embedding_lookup_2097: 
identity??embedding/embedding_lookup?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_2097string_lookup/Identity:output:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/2097*'
_output_shapes
:????????? *
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/2097*'
_output_shapes
:????????? ?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? }
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^embedding/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
*__inference_brute_force_layer_call_fn_1842
input_1
unknown
	unknown_0	
	unknown_1: 
	unknown_2:G 
	unknown_3:G
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_1827o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
C__inference_embedding_layer_call_and_return_conditional_losses_1714

inputs	'
embedding_lookup_1708: 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_1708inputs*
Tindices0	*(
_class
loc:@embedding_lookup/1708*'
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/1708*'
_output_shapes
:????????? }
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_2090

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	1
embedding_embedding_lookup_2084: 
identity??embedding/embedding_lookup?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_2084string_lookup/Identity:output:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/2084*'
_output_shapes
:????????? *
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/2084*'
_output_shapes
:????????? ?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? }
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^embedding/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_1728
string_lookup_input
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?	
?
*__inference_brute_force_layer_call_fn_1973
queries
unknown
	unknown_0	
	unknown_1: 
	unknown_2:G 
	unknown_3:G
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueriesunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????
:?????????
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_brute_force_layer_call_and_return_conditional_losses_1827o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	queries:

_output_shapes
: 
?
?
__inference__initializer_21325
1key_value_init70_lookuptableimportv2_table_handle-
)key_value_init70_lookuptableimportv2_keys/
+key_value_init70_lookuptableimportv2_values	
identity??$key_value_init70/LookupTableImportV2?
$key_value_init70/LookupTableImportV2LookupTableImportV21key_value_init70_lookuptableimportv2_table_handle)key_value_init70_lookuptableimportv2_keys+key_value_init70_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init70/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$key_value_init70/LookupTableImportV2$key_value_init70/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input_1,
serving_default_input_1:0?????????>
output_12
StatefulPartitionedCall_1:0?????????
>
output_22
StatefulPartitionedCall_1:1?????????
tensorflow/serving/predict:?M
?
query_model
identifiers
_identifiers

candidates
_candidates
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
query_with_exclusions

signatures"
_tf_keras_model
?
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
:G2identifiers
:G 2
candidates
5
0
1
2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_brute_force_layer_call_fn_1842
*__inference_brute_force_layer_call_fn_1973
*__inference_brute_force_layer_call_fn_1990
*__inference_brute_force_layer_call_fn_1914?
???
FullArgSpec/
args'?$
jself
	jqueries
jk

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_brute_force_layer_call_and_return_conditional_losses_2013
E__inference_brute_force_layer_call_and_return_conditional_losses_2036
E__inference_brute_force_layer_call_and_return_conditional_losses_1935
E__inference_brute_force_layer_call_and_return_conditional_losses_1956?
???
FullArgSpec/
args'?$
jself
	jqueries
jk

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference__wrapped_model_1694input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
	jqueries
j
exclusions
jk
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
serving_default"
signature_map
:
lookup_table
	keras_api"
_tf_keras_layer
?

embeddings
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_sequential_layer_call_fn_1728
)__inference_sequential_layer_call_fn_2066
)__inference_sequential_layer_call_fn_2077
)__inference_sequential_layer_call_fn_1780?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_2090
D__inference_sequential_layer_call_and_return_conditional_losses_2103
D__inference_sequential_layer_call_and_return_conditional_losses_1791
D__inference_sequential_layer_call_and_return_conditional_losses_1802?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
&:$ 2embedding/embeddings
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
"__inference_signature_wrapper_2055input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
j
)_initializer
*_create_resource
+_initialize
,_destroy_resourceR jCustom.StaticHashTable
"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_embedding_layer_call_fn_2110?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_embedding_layer_call_and_return_conditional_losses_2119?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
"
_generic_user_object
?2?
__inference__creator_2124?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_2132?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2137?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
	J
Const
J	
Const_1
J	
Const_25
__inference__creator_2124?

? 
? "? 7
__inference__destroyer_2137?

? 
? "? >
__inference__initializer_213234?

? 
? "? ?
__inference__wrapped_model_1694?2,?)
"?
?
input_1?????????
? "c?`
.
output_1"?
output_1?????????

.
output_2"?
output_2?????????
?
E__inference_brute_force_layer_call_and_return_conditional_losses_1935?24?1
*?'
?
input_1?????????

 
p 
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_1956?24?1
*?'
?
input_1?????????

 
p
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_2013?24?1
*?'
?
queries?????????

 
p 
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
E__inference_brute_force_layer_call_and_return_conditional_losses_2036?24?1
*?'
?
queries?????????

 
p
? "K?H
A?>
?
0/0?????????

?
0/1?????????

? ?
*__inference_brute_force_layer_call_fn_1842|24?1
*?'
?
input_1?????????

 
p 
? "=?:
?
0?????????

?
1?????????
?
*__inference_brute_force_layer_call_fn_1914|24?1
*?'
?
input_1?????????

 
p
? "=?:
?
0?????????

?
1?????????
?
*__inference_brute_force_layer_call_fn_1973|24?1
*?'
?
queries?????????

 
p 
? "=?:
?
0?????????

?
1?????????
?
*__inference_brute_force_layer_call_fn_1990|24?1
*?'
?
queries?????????

 
p
? "=?:
?
0?????????

?
1?????????
?
C__inference_embedding_layer_call_and_return_conditional_losses_2119W+?(
!?
?
inputs?????????	
? "%?"
?
0????????? 
? v
(__inference_embedding_layer_call_fn_2110J+?(
!?
?
inputs?????????	
? "?????????? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1791n2@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "%?"
?
0????????? 
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1802n2@?=
6?3
)?&
string_lookup_input?????????
p

 
? "%?"
?
0????????? 
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_2090a23?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_2103a23?0
)?&
?
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
)__inference_sequential_layer_call_fn_1728a2@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "?????????? ?
)__inference_sequential_layer_call_fn_1780a2@?=
6?3
)?&
string_lookup_input?????????
p

 
? "?????????? ?
)__inference_sequential_layer_call_fn_2066T23?0
)?&
?
inputs?????????
p 

 
? "?????????? ?
)__inference_sequential_layer_call_fn_2077T23?0
)?&
?
inputs?????????
p

 
? "?????????? ?
"__inference_signature_wrapper_2055?27?4
? 
-?*
(
input_1?
input_1?????????"c?`
.
output_1"?
output_1?????????

.
output_2"?
output_2?????????
