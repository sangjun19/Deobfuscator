	.file	"FrameitMeZ_FirstMeetTeam_area_choice_flatten.c"
	.text
	.globl	_TIG_IZ_dF1l_argc
	.bss
	.align 4
	.type	_TIG_IZ_dF1l_argc, @object
	.size	_TIG_IZ_dF1l_argc, 4
_TIG_IZ_dF1l_argc:
	.zero	4
	.globl	_TIG_IZ_dF1l_envp
	.align 8
	.type	_TIG_IZ_dF1l_envp, @object
	.size	_TIG_IZ_dF1l_envp, 8
_TIG_IZ_dF1l_envp:
	.zero	8
	.globl	_TIG_IZ_dF1l_argv
	.align 8
	.type	_TIG_IZ_dF1l_argv, @object
	.size	_TIG_IZ_dF1l_argv, 8
_TIG_IZ_dF1l_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"The area of the triangle is: %.2f"
	.align 8
.LC2:
	.string	"The area of the triangle is: Error"
	.align 8
.LC3:
	.string	"The area of the rectangle is: Error"
.LC4:
	.string	"Enter the base: "
.LC5:
	.string	"%f"
.LC6:
	.string	"Enter the height: "
.LC8:
	.string	"Enter the width: "
.LC9:
	.string	"Enter the radius: "
	.align 8
.LC11:
	.string	"The area of the circle is: Error"
	.align 8
.LC12:
	.string	"1. Rectangle\n2. Triangle\n3. Circle\nEnter your choice: "
.LC13:
	.string	"%d"
	.align 8
.LC14:
	.string	"The area of the rectangle is: %.2f"
	.align 8
.LC15:
	.string	"The area of the circle is: %.2f"
.LC16:
	.string	"Invalid choice"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_dF1l_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_dF1l_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_dF1l_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-dF1l--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_dF1l_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_dF1l_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_dF1l_envp(%rip)
	nop
	movq	$20, -16(%rbp)
.L61:
	cmpq	$31, -16(%rbp)
	ja	.L74
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
	.long	.L74-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L74-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L74-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L74-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L74-.L8
	.long	.L11-.L8
	.long	.L74-.L8
	.long	.L74-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L29:
	pxor	%xmm2, %xmm2
	cvtss2sd	-24(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L9:
	movl	-48(%rbp), %eax
	cmpl	$3, %eax
	je	.L34
	cmpl	$3, %eax
	jg	.L35
	cmpl	$1, %eax
	je	.L36
	cmpl	$2, %eax
	je	.L37
	jmp	.L35
.L34:
	movq	$13, -16(%rbp)
	jmp	.L38
.L37:
	movq	$26, -16(%rbp)
	jmp	.L38
.L36:
	movq	$11, -16(%rbp)
	jmp	.L38
.L35:
	movq	$9, -16(%rbp)
	nop
.L38:
	jmp	.L33
.L20:
	movl	-48(%rbp), %eax
	cmpl	$3, %eax
	jg	.L39
	movq	$24, -16(%rbp)
	jmp	.L33
.L39:
	movq	$15, -16(%rbp)
	jmp	.L33
.L19:
	movl	-48(%rbp), %eax
	cmpl	$3, %eax
	jle	.L41
	movq	$2, -16(%rbp)
	jmp	.L33
.L41:
	movq	$6, -16(%rbp)
	jmp	.L33
.L7:
	movss	-40(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L68
	movq	$5, -16(%rbp)
	jmp	.L33
.L68:
	movq	$10, -16(%rbp)
	jmp	.L33
.L22:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L32:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L13:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L30:
	movss	-36(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L69
	movq	$12, -16(%rbp)
	jmp	.L33
.L69:
	movq	$17, -16(%rbp)
	jmp	.L33
.L18:
	movss	-28(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L70
	movq	$19, -16(%rbp)
	jmp	.L33
.L70:
	movq	$7, -16(%rbp)
	jmp	.L33
.L12:
	movl	-48(%rbp), %eax
	testl	%eax, %eax
	jle	.L52
	movq	$30, -16(%rbp)
	jmp	.L33
.L52:
	movq	$15, -16(%rbp)
	jmp	.L33
.L11:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movss	-36(%rbp), %xmm1
	movss	-32(%rbp), %xmm0
	mulss	%xmm0, %xmm1
	movss	.LC7(%rip), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L33
.L23:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$29, -16(%rbp)
	jmp	.L33
.L25:
	movq	$6, -16(%rbp)
	jmp	.L33
.L21:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movss	-28(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movsd	.LC10(%rip), %xmm0
	mulsd	%xmm0, %xmm1
	movss	-28(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	mulsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L33
.L16:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L17:
	movss	-32(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L71
	movq	$1, -16(%rbp)
	jmp	.L33
.L71:
	movq	$4, -16(%rbp)
	jmp	.L33
.L27:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L62
	jmp	.L72
.L14:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$14, -16(%rbp)
	jmp	.L33
.L28:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L24:
	movss	-44(%rbp), %xmm1
	movss	-40(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	pxor	%xmm3, %xmm3
	cvtss2sd	%xmm0, %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L26:
	pxor	%xmm4, %xmm4
	cvtss2sd	-20(%rbp), %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L10:
	movss	-44(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L73
	movq	$23, -16(%rbp)
	jmp	.L33
.L73:
	movq	$31, -16(%rbp)
	jmp	.L33
.L31:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L33
.L15:
	movq	$22, -16(%rbp)
	jmp	.L33
.L74:
	nop
.L33:
	jmp	.L61
.L72:
	call	__stack_chk_fail@PLT
.L62:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC7:
	.long	1056964608
	.align 8
.LC10:
	.long	1374389535
	.long	1074339512
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
