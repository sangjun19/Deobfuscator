	.file	"eeeeeeeeeeekk_e-for-dinosaurs_c_flatten.c"
	.text
	.globl	_TIG_IZ_oRvd_argc
	.bss
	.align 4
	.type	_TIG_IZ_oRvd_argc, @object
	.size	_TIG_IZ_oRvd_argc, 4
_TIG_IZ_oRvd_argc:
	.zero	4
	.globl	_TIG_IZ_oRvd_envp
	.align 8
	.type	_TIG_IZ_oRvd_envp, @object
	.size	_TIG_IZ_oRvd_envp, 8
_TIG_IZ_oRvd_envp:
	.zero	8
	.globl	_TIG_IZ_oRvd_argv
	.align 8
	.type	_TIG_IZ_oRvd_argv, @object
	.size	_TIG_IZ_oRvd_argv, 8
_TIG_IZ_oRvd_argv:
	.zero	8
	.section	.rodata
.LC1:
	.string	"e = %lf\n"
.LC4:
	.string	"Number of iterations: "
.LC5:
	.string	"%llu"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_oRvd_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_oRvd_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_oRvd_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 116 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-oRvd--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_oRvd_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_oRvd_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_oRvd_envp(%rip)
	nop
	movq	$8, -24(%rbp)
.L24:
	cmpq	$13, -24(%rbp)
	ja	.L29
	movq	-24(%rbp), %rax
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
	.long	.L29-.L8
	.long	.L16-.L8
	.long	.L29-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L29-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L29-.L8
	.long	.L29-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L25
	jmp	.L27
.L9:
	movss	.LC0(%rip), %xmm0
	comiss	-56(%rbp), %xmm0
	jbe	.L28
	movq	$3, -24(%rbp)
	jmp	.L21
.L28:
	movq	$9, -24(%rbp)
	jmp	.L21
.L11:
	movq	$7, -24(%rbp)
	jmp	.L21
.L16:
	pxor	%xmm0, %xmm0
	cvtsi2sdl	-60(%rbp), %xmm0
	movq	-48(%rbp), %rax
	pxor	%xmm1, %xmm1
	cvtsi2sdq	%rax, %xmm1
	divsd	%xmm1, %xmm0
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$4, -24(%rbp)
	jmp	.L21
.L15:
	call	rand@PLT
	movl	%eax, -52(%rbp)
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-52(%rbp), %xmm0
	movss	.LC2(%rip), %xmm1
	divss	%xmm1, %xmm0
	movss	-56(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -56(%rbp)
	addl	$1, -60(%rbp)
	movq	$12, -24(%rbp)
	jmp	.L21
.L10:
	addq	$1, -32(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L21
.L7:
	movq	-48(%rbp), %rax
	cmpq	%rax, -32(%rbp)
	jge	.L22
	movq	$5, -24(%rbp)
	jmp	.L21
.L22:
	movq	$1, -24(%rbp)
	jmp	.L21
.L13:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -56(%rbp)
	movq	$12, -24(%rbp)
	jmp	.L21
.L12:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	time@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, %edi
	call	srand@PLT
	movl	$0, -60(%rbp)
	movq	$0, -32(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L21
.L29:
	nop
.L21:
	jmp	.L24
.L27:
	call	__stack_chk_fail@PLT
.L25:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC0:
	.long	1065353216
	.align 4
.LC2:
	.long	1325400064
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
