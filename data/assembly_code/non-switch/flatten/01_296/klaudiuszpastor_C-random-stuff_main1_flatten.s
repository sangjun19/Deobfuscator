	.file	"klaudiuszpastor_C-random-stuff_main1_flatten.c"
	.text
	.globl	_TIG_IZ_w0Jl_argc
	.bss
	.align 4
	.type	_TIG_IZ_w0Jl_argc, @object
	.size	_TIG_IZ_w0Jl_argc, 4
_TIG_IZ_w0Jl_argc:
	.zero	4
	.globl	_TIG_IZ_w0Jl_envp
	.align 8
	.type	_TIG_IZ_w0Jl_envp, @object
	.size	_TIG_IZ_w0Jl_envp, 8
_TIG_IZ_w0Jl_envp:
	.zero	8
	.globl	_TIG_IZ_w0Jl_argv
	.align 8
	.type	_TIG_IZ_w0Jl_argv, @object
	.size	_TIG_IZ_w0Jl_argv, 8
_TIG_IZ_w0Jl_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Values of a is %d"
	.text
	.globl	fun
	.type	fun, @function
fun:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	fun, .-fun
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_w0Jl_envp(%rip)
	nop
.L10:
	movq	$0, _TIG_IZ_w0Jl_argv(%rip)
	nop
.L11:
	movl	$0, _TIG_IZ_w0Jl_argc(%rip)
	nop
	nop
.L12:
.L13:
#APP
# 100 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-w0Jl--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_w0Jl_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_w0Jl_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_w0Jl_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L19:
	cmpq	$2, -16(%rbp)
	je	.L14
	cmpq	$2, -16(%rbp)
	ja	.L21
	cmpq	$0, -16(%rbp)
	je	.L16
	cmpq	$1, -16(%rbp)
	jne	.L21
	movq	$0, -16(%rbp)
	jmp	.L17
.L16:
	leaq	fun(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	$10, %edi
	call	*%rax
	movq	$2, -16(%rbp)
	jmp	.L17
.L14:
	movl	$0, %eax
	jmp	.L20
.L21:
	nop
.L17:
	jmp	.L19
.L20:
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
