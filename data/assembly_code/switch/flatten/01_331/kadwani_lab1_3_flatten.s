	.file	"kadwani_lab1_3_flatten.c"
	.text
	.globl	_TIG_IZ_lGR6_envp
	.bss
	.align 8
	.type	_TIG_IZ_lGR6_envp, @object
	.size	_TIG_IZ_lGR6_envp, 8
_TIG_IZ_lGR6_envp:
	.zero	8
	.globl	_TIG_IZ_lGR6_argc
	.align 4
	.type	_TIG_IZ_lGR6_argc, @object
	.size	_TIG_IZ_lGR6_argc, 4
_TIG_IZ_lGR6_argc:
	.zero	4
	.globl	_TIG_IZ_lGR6_argv
	.align 8
	.type	_TIG_IZ_lGR6_argv, @object
	.size	_TIG_IZ_lGR6_argv, 8
_TIG_IZ_lGR6_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%.9g%c%.9g =  %.6g\n\n"
.LC1:
	.string	"Enter calculation:\n"
.LC2:
	.string	"%f %c %f"
.LC3:
	.string	"Fail."
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_lGR6_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_lGR6_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_lGR6_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-lGR6--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_lGR6_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_lGR6_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_lGR6_envp(%rip)
	nop
	movq	$21, -32(%rbp)
.L32:
	cmpq	$21, -32(%rbp)
	ja	.L35
	movq	-32(%rbp), %rax
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
	.long	.L19-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L35-.L8
	.long	.L16-.L8
	.long	.L35-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L12-.L8
	.long	.L35-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L35-.L8
	.long	.L9-.L8
	.long	.L35-.L8
	.long	.L7-.L8
	.text
.L17:
	movss	-40(%rbp), %xmm0
	pxor	%xmm3, %xmm3
	cvtss2sd	%xmm0, %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	pxor	%xmm0, %xmm0
	cvtsd2ss	-16(%rbp), %xmm0
	movss	%xmm0, -36(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L20
.L12:
	pxor	%xmm1, %xmm1
	cvtss2sd	-36(%rbp), %xmm1
	movss	-40(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movzbl	-45(%rbp), %eax
	movsbl	%al, %edx
	movss	-44(%rbp), %xmm2
	pxor	%xmm4, %xmm4
	cvtss2sd	%xmm2, %xmm4
	movq	%xmm4, %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movl	%edx, %esi
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$19, -32(%rbp)
	jmp	.L20
.L15:
	movss	-44(%rbp), %xmm1
	movss	-40(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -36(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L20
.L18:
	movss	-44(%rbp), %xmm1
	movss	-40(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -36(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L20
.L11:
	movss	-40(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-44(%rbp), %xmm1
	pxor	%xmm5, %xmm5
	cvtss2sd	%xmm1, %xmm5
	movq	%xmm5, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	pxor	%xmm0, %xmm0
	cvtsd2ss	-24(%rbp), %xmm0
	movss	%xmm0, -36(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L20
.L7:
	movq	$17, -32(%rbp)
	jmp	.L20
.L14:
	movss	-44(%rbp), %xmm0
	movss	-40(%rbp), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -36(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L20
.L9:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-40(%rbp), %rcx
	leaq	-45(%rbp), %rdx
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -32(%rbp)
	jmp	.L20
.L16:
	movzbl	-45(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	jg	.L22
	cmpl	$32, %eax
	jl	.L23
	subl	$32, %eax
	cmpl	$15, %eax
	ja	.L23
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L29-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L23-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L23-.L25
	.long	.L26-.L25
	.long	.L23-.L25
	.long	.L24-.L25
	.text
.L22:
	cmpl	$94, %eax
	je	.L30
	jmp	.L23
.L29:
	movq	$4, -32(%rbp)
	jmp	.L31
.L30:
	movq	$16, -32(%rbp)
	jmp	.L31
.L26:
	movq	$9, -32(%rbp)
	jmp	.L31
.L27:
	movq	$3, -32(%rbp)
	jmp	.L31
.L28:
	movq	$8, -32(%rbp)
	jmp	.L31
.L24:
	movq	$0, -32(%rbp)
	jmp	.L31
.L23:
	movq	$10, -32(%rbp)
	nop
.L31:
	jmp	.L20
.L13:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -32(%rbp)
	jmp	.L20
.L19:
	movss	-44(%rbp), %xmm0
	movss	-40(%rbp), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -36(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L20
.L35:
	nop
.L20:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
