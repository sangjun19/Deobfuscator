	.file	"flaviogf_courses_app_flatten.c"
	.text
	.globl	_TIG_IZ_HgyS_envp
	.bss
	.align 8
	.type	_TIG_IZ_HgyS_envp, @object
	.size	_TIG_IZ_HgyS_envp, 8
_TIG_IZ_HgyS_envp:
	.zero	8
	.globl	_TIG_IZ_HgyS_argv
	.align 8
	.type	_TIG_IZ_HgyS_argv, @object
	.size	_TIG_IZ_HgyS_argv, 8
_TIG_IZ_HgyS_argv:
	.zero	8
	.globl	_TIG_IZ_HgyS_argc
	.align 4
	.type	_TIG_IZ_HgyS_argc, @object
	.size	_TIG_IZ_HgyS_argc, 4
_TIG_IZ_HgyS_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Oops, this is not a valid expression"
.LC1:
	.string	"Type in your expression."
.LC2:
	.string	"%f %c %f"
.LC3:
	.string	"%.2f\n"
	.text
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_HgyS_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_HgyS_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_HgyS_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-HgyS--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_HgyS_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_HgyS_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_HgyS_envp(%rip)
	nop
	movq	$10, -16(%rbp)
.L25:
	cmpq	$12, -16(%rbp)
	ja	.L28
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
	.long	.L16-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L28-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L7:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-20(%rbp), %rcx
	leaq	-25(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -16(%rbp)
	jmp	.L17
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L9:
	movss	-24(%rbp), %xmm0
	movss	-20(%rbp), %xmm1
	subss	%xmm1, %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	%xmm0, %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L13:
	movzbl	-25(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L19
	cmpl	$47, %eax
	jg	.L20
	cmpl	$45, %eax
	je	.L21
	cmpl	$45, %eax
	jg	.L20
	cmpl	$42, %eax
	je	.L22
	cmpl	$43, %eax
	je	.L23
	jmp	.L20
.L19:
	movq	$7, -16(%rbp)
	jmp	.L24
.L22:
	movq	$0, -16(%rbp)
	jmp	.L24
.L21:
	movq	$11, -16(%rbp)
	jmp	.L24
.L23:
	movq	$5, -16(%rbp)
	jmp	.L24
.L20:
	movq	$4, -16(%rbp)
	nop
.L24:
	jmp	.L17
.L14:
	movss	-24(%rbp), %xmm1
	movss	-20(%rbp), %xmm0
	addss	%xmm1, %xmm0
	pxor	%xmm3, %xmm3
	cvtss2sd	%xmm0, %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L10:
	movq	$12, -16(%rbp)
	jmp	.L17
.L16:
	movss	-24(%rbp), %xmm1
	movss	-20(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	pxor	%xmm4, %xmm4
	cvtss2sd	%xmm0, %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L12:
	movss	-24(%rbp), %xmm0
	movss	-20(%rbp), %xmm1
	divss	%xmm1, %xmm0
	pxor	%xmm5, %xmm5
	cvtss2sd	%xmm0, %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L17
.L28:
	nop
.L17:
	jmp	.L25
.L27:
	call	__stack_chk_fail@PLT
.L26:
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
