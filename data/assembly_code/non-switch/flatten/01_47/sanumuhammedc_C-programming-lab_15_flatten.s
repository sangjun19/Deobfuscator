	.file	"sanumuhammedc_C-programming-lab_15_flatten.c"
	.text
	.globl	_TIG_IZ_vDpK_argv
	.bss
	.align 8
	.type	_TIG_IZ_vDpK_argv, @object
	.size	_TIG_IZ_vDpK_argv, 8
_TIG_IZ_vDpK_argv:
	.zero	8
	.globl	_TIG_IZ_vDpK_argc
	.align 4
	.type	_TIG_IZ_vDpK_argc, @object
	.size	_TIG_IZ_vDpK_argc, 4
_TIG_IZ_vDpK_argc:
	.zero	4
	.globl	_TIG_IZ_vDpK_envp
	.align 8
	.type	_TIG_IZ_vDpK_envp, @object
	.size	_TIG_IZ_vDpK_envp, 8
_TIG_IZ_vDpK_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the number of terms to be printed: "
.LC1:
	.string	"%d"
.LC2:
	.string	"\nThe series is as follows:"
.LC3:
	.string	"%d "
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_vDpK_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_vDpK_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_vDpK_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 102 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-vDpK--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_vDpK_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_vDpK_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_vDpK_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L17:
	cmpq	$6, -16(%rbp)
	ja	.L20
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L20-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L20-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L18
	jmp	.L19
.L12:
	movl	-28(%rbp), %eax
	cmpl	%eax, -24(%rbp)
	jg	.L14
	movq	$2, -16(%rbp)
	jmp	.L16
.L14:
	movq	$4, -16(%rbp)
	jmp	.L16
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L16
.L9:
	movq	$6, -16(%rbp)
	jmp	.L16
.L11:
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	fib
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L16
.L20:
	nop
.L16:
	jmp	.L17
.L19:
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	fib
	.type	fib, @function
fib:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$2, -8(%rbp)
.L31:
	cmpq	$5, -8(%rbp)
	je	.L22
	cmpq	$5, -8(%rbp)
	ja	.L32
	cmpq	$3, -8(%rbp)
	je	.L24
	cmpq	$3, -8(%rbp)
	ja	.L32
	cmpq	$1, -8(%rbp)
	je	.L25
	cmpq	$2, -8(%rbp)
	je	.L26
	jmp	.L32
.L25:
	movl	-20(%rbp), %eax
	jmp	.L27
.L24:
	movl	-20(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %edi
	call	fib
	movl	%eax, -16(%rbp)
	movl	-20(%rbp), %eax
	subl	$2, %eax
	movl	%eax, %edi
	call	fib
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L28
.L22:
	movl	-16(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	jmp	.L27
.L26:
	cmpl	$1, -20(%rbp)
	jg	.L29
	movq	$1, -8(%rbp)
	jmp	.L28
.L29:
	movq	$3, -8(%rbp)
	jmp	.L28
.L32:
	nop
.L28:
	jmp	.L31
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	fib, .-fib
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
