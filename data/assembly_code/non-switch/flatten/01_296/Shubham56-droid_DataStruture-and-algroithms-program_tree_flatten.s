	.file	"Shubham56-droid_DataStruture-and-algroithms-program_tree_flatten.c"
	.text
	.globl	_TIG_IZ_oUot_envp
	.bss
	.align 8
	.type	_TIG_IZ_oUot_envp, @object
	.size	_TIG_IZ_oUot_envp, 8
_TIG_IZ_oUot_envp:
	.zero	8
	.globl	_TIG_IZ_oUot_argc
	.align 4
	.type	_TIG_IZ_oUot_argc, @object
	.size	_TIG_IZ_oUot_argc, 4
_TIG_IZ_oUot_argc:
	.zero	4
	.globl	_TIG_IZ_oUot_argv
	.align 8
	.type	_TIG_IZ_oUot_argv, @object
	.size	_TIG_IZ_oUot_argv, 8
_TIG_IZ_oUot_argv:
	.zero	8
	.text
	.globl	newNode
	.type	newNode, @function
newNode:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$2, -16(%rbp)
.L7:
	cmpq	$2, -16(%rbp)
	je	.L2
	cmpq	$2, -16(%rbp)
	ja	.L9
	cmpq	$0, -16(%rbp)
	je	.L4
	cmpq	$1, -16(%rbp)
	jne	.L9
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$0, -16(%rbp)
	jmp	.L5
.L4:
	movq	-24(%rbp), %rax
	jmp	.L8
.L2:
	movq	$1, -16(%rbp)
	jmp	.L5
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	newNode, .-newNode
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$72, %rsp
	.cfi_offset 3, -24
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$0, _TIG_IZ_oUot_envp(%rip)
	nop
.L11:
	movq	$0, _TIG_IZ_oUot_argv(%rip)
	nop
.L12:
	movl	$0, _TIG_IZ_oUot_argc(%rip)
	nop
	nop
.L13:
.L14:
#APP
# 81 "Shubham56-droid_DataStruture-and-algroithms-program_tree.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-oUot--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_oUot_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_oUot_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_oUot_envp(%rip)
	nop
	movq	$1, -40(%rbp)
.L20:
	cmpq	$2, -40(%rbp)
	je	.L15
	cmpq	$2, -40(%rbp)
	ja	.L22
	cmpq	$0, -40(%rbp)
	je	.L17
	cmpq	$1, -40(%rbp)
	jne	.L22
	movq	$2, -40(%rbp)
	jmp	.L18
.L17:
	movl	$0, %eax
	jmp	.L21
.L15:
	movl	$1, %edi
	call	newNode
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movl	$2, %edi
	call	newNode
	movq	-24(%rbp), %rdx
	movq	%rax, 8(%rdx)
	movl	$3, %edi
	call	newNode
	movq	-24(%rbp), %rdx
	movq	%rax, 16(%rdx)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rbx
	movl	$4, %edi
	call	newNode
	movq	%rax, 8(%rbx)
	call	getchar@PLT
	movq	$0, -40(%rbp)
	jmp	.L18
.L22:
	nop
.L18:
	jmp	.L20
.L21:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
