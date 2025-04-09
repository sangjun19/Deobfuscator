	.file	"ericksjp_Algoritmos-C_ex186_flatten.c"
	.text
	.globl	_TIG_IZ_eBax_argv
	.bss
	.align 8
	.type	_TIG_IZ_eBax_argv, @object
	.size	_TIG_IZ_eBax_argv, 8
_TIG_IZ_eBax_argv:
	.zero	8
	.globl	_TIG_IZ_eBax_argc
	.align 4
	.type	_TIG_IZ_eBax_argc, @object
	.size	_TIG_IZ_eBax_argc, 4
_TIG_IZ_eBax_argc:
	.zero	4
	.globl	_TIG_IZ_eBax_envp
	.align 8
	.type	_TIG_IZ_eBax_envp, @object
	.size	_TIG_IZ_eBax_envp, 8
_TIG_IZ_eBax_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Informe um numero inteiro de 1 a 5 -> "
.LC1:
	.string	"%d"
	.align 8
.LC2:
	.string	"Informe um numero real da linha %d e coluna %d da matriz -> "
.LC3:
	.string	"%f"
	.align 8
.LC4:
	.string	"O maior elemento da linha %d da matriz eh: %f\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_eBax_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_eBax_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_eBax_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-eBax--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_eBax_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_eBax_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_eBax_envp(%rip)
	nop
	movq	$21, -120(%rbp)
.L32:
	cmpq	$21, -120(%rbp)
	ja	.L37
	movq	-120(%rbp), %rax
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
	.long	.L37-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L37-.L8
	.long	.L18-.L8
	.long	.L37-.L8
	.long	.L17-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L37-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L37-.L8
	.long	.L7-.L8
	.text
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-140(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-140(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -140(%rbp)
	movss	-112(%rbp), %xmm0
	movss	%xmm0, -128(%rbp)
	movl	$0, -124(%rbp)
	movq	$4, -120(%rbp)
	jmp	.L21
.L18:
	cmpl	$4, -124(%rbp)
	jg	.L22
	movq	$11, -120(%rbp)
	jmp	.L21
.L22:
	movq	$10, -120(%rbp)
	jmp	.L21
.L12:
	movl	-140(%rbp), %eax
	movl	-124(%rbp), %edx
	movslq	%edx, %rcx
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rcx, %rax
	movss	-112(%rbp,%rax,4), %xmm0
	movss	%xmm0, -128(%rbp)
	movq	$2, -120(%rbp)
	jmp	.L21
.L11:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -136(%rbp)
	movq	$6, -120(%rbp)
	jmp	.L21
.L13:
	cmpl	$4, -132(%rbp)
	jg	.L24
	movq	$9, -120(%rbp)
	jmp	.L21
.L24:
	movq	$15, -120(%rbp)
	jmp	.L21
.L20:
	movl	$0, -132(%rbp)
	movq	$12, -120(%rbp)
	jmp	.L21
.L7:
	movl	$0, -136(%rbp)
	movq	$6, -120(%rbp)
	jmp	.L21
.L14:
	movl	-140(%rbp), %eax
	movl	-124(%rbp), %edx
	movslq	%edx, %rcx
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rcx, %rax
	movss	-112(%rbp,%rax,4), %xmm0
	comiss	-128(%rbp), %xmm0
	jb	.L35
	movq	$14, -120(%rbp)
	jmp	.L21
.L35:
	movq	$2, -120(%rbp)
	jmp	.L21
.L16:
	movl	-132(%rbp), %eax
	leal	1(%rax), %edx
	movl	-136(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rcx
	movl	-132(%rbp), %eax
	movslq	%eax, %rsi
	movl	-136(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rsi, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -132(%rbp)
	movq	$12, -120(%rbp)
	jmp	.L21
.L9:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L36
.L17:
	cmpl	$4, -136(%rbp)
	jg	.L30
	movq	$1, -120(%rbp)
	jmp	.L21
.L30:
	movq	$18, -120(%rbp)
	jmp	.L21
.L15:
	pxor	%xmm1, %xmm1
	cvtss2sd	-128(%rbp), %xmm1
	movq	%xmm1, %rdx
	movl	-140(%rbp), %eax
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$19, -120(%rbp)
	jmp	.L21
.L19:
	addl	$1, -124(%rbp)
	movq	$4, -120(%rbp)
	jmp	.L21
.L37:
	nop
.L21:
	jmp	.L32
.L36:
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
