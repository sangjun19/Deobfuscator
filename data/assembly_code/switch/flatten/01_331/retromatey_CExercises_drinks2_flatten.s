	.file	"retromatey_CExercises_drinks2_flatten.c"
	.text
	.globl	_TIG_IZ_iGva_argc
	.bss
	.align 4
	.type	_TIG_IZ_iGva_argc, @object
	.size	_TIG_IZ_iGva_argc, 4
_TIG_IZ_iGva_argc:
	.zero	4
	.globl	_TIG_IZ_iGva_envp
	.align 8
	.type	_TIG_IZ_iGva_envp, @object
	.size	_TIG_IZ_iGva_envp, 8
_TIG_IZ_iGva_envp:
	.zero	8
	.globl	_TIG_IZ_iGva_argv
	.align 8
	.type	_TIG_IZ_iGva_argv, @object
	.size	_TIG_IZ_iGva_argv, 8
_TIG_IZ_iGva_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"The price is %.2f\n"
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
	movq	$0, _TIG_IZ_iGva_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_iGva_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_iGva_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-iGva--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_iGva_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_iGva_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_iGva_envp(%rip)
	nop
	movq	$1, -32(%rbp)
.L11:
	cmpq	$2, -32(%rbp)
	je	.L6
	cmpq	$2, -32(%rbp)
	ja	.L13
	cmpq	$0, -32(%rbp)
	je	.L8
	cmpq	$1, -32(%rbp)
	jne	.L13
	movq	$0, -32(%rbp)
	jmp	.L9
.L8:
	movl	$0, %edx
	movl	$2, %esi
	movl	$2, %edi
	movl	$0, %eax
	call	total
	movq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movl	$1, %ecx
	movl	$0, %edx
	movl	$2, %esi
	movl	$3, %edi
	movl	$0, %eax
	call	total
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movl	$3, %esi
	movl	$1, %edi
	movl	$0, %eax
	call	total
	movq	%xmm0, %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -32(%rbp)
	jmp	.L9
.L6:
	movl	$0, %eax
	jmp	.L12
.L13:
	nop
.L9:
	jmp	.L11
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	total
	.type	total, @function
total:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$272, %rsp
	movl	%edi, -260(%rbp)
	movq	%rsi, -168(%rbp)
	movq	%rdx, -160(%rbp)
	movq	%rcx, -152(%rbp)
	movq	%r8, -144(%rbp)
	movq	%r9, -136(%rbp)
	testb	%al, %al
	je	.L15
	movaps	%xmm0, -128(%rbp)
	movaps	%xmm1, -112(%rbp)
	movaps	%xmm2, -96(%rbp)
	movaps	%xmm3, -80(%rbp)
	movaps	%xmm4, -64(%rbp)
	movaps	%xmm5, -48(%rbp)
	movaps	%xmm6, -32(%rbp)
	movaps	%xmm7, -16(%rbp)
.L15:
	movq	%fs:40, %rax
	movq	%rax, -184(%rbp)
	xorl	%eax, %eax
	movq	$5, -224(%rbp)
.L30:
	cmpq	$7, -224(%rbp)
	ja	.L33
	movq	-224(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L18(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L18(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L18:
	.long	.L23-.L18
	.long	.L22-.L18
	.long	.L33-.L18
	.long	.L33-.L18
	.long	.L21-.L18
	.long	.L20-.L18
	.long	.L19-.L18
	.long	.L17-.L18
	.text
.L21:
	movl	-208(%rbp), %eax
	cmpl	$47, %eax
	ja	.L24
	movq	-192(%rbp), %rax
	movl	-208(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-208(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -208(%rbp)
	jmp	.L25
.L24:
	movq	-200(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -200(%rbp)
.L25:
	movl	(%rax), %eax
	movl	%eax, -240(%rbp)
	movl	-240(%rbp), %eax
	movl	%eax, -236(%rbp)
	movl	-236(%rbp), %eax
	movl	%eax, %edi
	call	price
	movq	%xmm0, %rax
	movq	%rax, -216(%rbp)
	movsd	-232(%rbp), %xmm0
	addsd	-216(%rbp), %xmm0
	movsd	%xmm0, -232(%rbp)
	addl	$1, -244(%rbp)
	movq	$6, -224(%rbp)
	jmp	.L26
.L22:
	movsd	-232(%rbp), %xmm0
	movq	%xmm0, %rax
	movq	-184(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	jmp	.L32
.L19:
	movl	-244(%rbp), %eax
	cmpl	-260(%rbp), %eax
	jge	.L28
	movq	$4, -224(%rbp)
	jmp	.L26
.L28:
	movq	$0, -224(%rbp)
	jmp	.L26
.L20:
	movq	$7, -224(%rbp)
	jmp	.L26
.L23:
	movq	$1, -224(%rbp)
	jmp	.L26
.L17:
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -232(%rbp)
	movl	$8, -208(%rbp)
	movl	$48, -204(%rbp)
	leaq	16(%rbp), %rax
	movq	%rax, -200(%rbp)
	leaq	-176(%rbp), %rax
	movq	%rax, -192(%rbp)
	movl	$0, -244(%rbp)
	movq	$6, -224(%rbp)
	jmp	.L26
.L33:
	nop
.L26:
	jmp	.L30
.L32:
	call	__stack_chk_fail@PLT
.L31:
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	total, .-total
	.globl	price
	.type	price, @function
price:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$4, -8(%rbp)
.L52:
	cmpq	$6, -8(%rbp)
	ja	.L53
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L37(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L37(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L37:
	.long	.L43-.L37
	.long	.L42-.L37
	.long	.L41-.L37
	.long	.L40-.L37
	.long	.L39-.L37
	.long	.L38-.L37
	.long	.L36-.L37
	.text
.L39:
	cmpl	$3, -20(%rbp)
	je	.L44
	cmpl	$3, -20(%rbp)
	ja	.L45
	cmpl	$2, -20(%rbp)
	je	.L46
	cmpl	$2, -20(%rbp)
	ja	.L45
	cmpl	$0, -20(%rbp)
	je	.L47
	cmpl	$1, -20(%rbp)
	je	.L48
	jmp	.L45
.L44:
	movq	$0, -8(%rbp)
	jmp	.L49
.L46:
	movq	$1, -8(%rbp)
	jmp	.L49
.L48:
	movq	$6, -8(%rbp)
	jmp	.L49
.L47:
	movq	$5, -8(%rbp)
	jmp	.L49
.L45:
	movq	$3, -8(%rbp)
	nop
.L49:
	jmp	.L50
.L42:
	movsd	.LC2(%rip), %xmm0
	jmp	.L51
.L40:
	movq	$2, -8(%rbp)
	jmp	.L50
.L36:
	movsd	.LC3(%rip), %xmm0
	jmp	.L51
.L38:
	movsd	.LC4(%rip), %xmm0
	jmp	.L51
.L43:
	movsd	.LC5(%rip), %xmm0
	jmp	.L51
.L41:
	pxor	%xmm0, %xmm0
	jmp	.L51
.L53:
	nop
.L50:
	jmp	.L52
.L51:
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	price, .-price
	.section	.rodata
	.align 8
.LC2:
	.long	343597384
	.long	1075005358
	.align 8
.LC3:
	.long	-1546188227
	.long	1075133808
	.align 8
.LC4:
	.long	-1030792151
	.long	1075521781
	.align 8
.LC5:
	.long	687194767
	.long	1075285852
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
