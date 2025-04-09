	.file	"Creyving5_CreyvingGuerrero_PM_main_flatten.c"
	.text
	.globl	_TIG_IZ_yYFX_envp
	.bss
	.align 8
	.type	_TIG_IZ_yYFX_envp, @object
	.size	_TIG_IZ_yYFX_envp, 8
_TIG_IZ_yYFX_envp:
	.zero	8
	.globl	_TIG_IZ_yYFX_argc
	.align 4
	.type	_TIG_IZ_yYFX_argc, @object
	.size	_TIG_IZ_yYFX_argc, 4
_TIG_IZ_yYFX_argc:
	.zero	4
	.globl	_TIG_IZ_yYFX_argv
	.align 8
	.type	_TIG_IZ_yYFX_argv, @object
	.size	_TIG_IZ_yYFX_argv, 8
_TIG_IZ_yYFX_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC5:
	.string	"Ingresa la clave  y el tiempo: "
.LC6:
	.string	"%d %d"
	.align 8
.LC8:
	.string	"\n\nClave: %d\tTiempo: %d\tCosto: %6.2f"
.LC10:
	.string	"\nError en la clave"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_yYFX_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_yYFX_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_yYFX_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-yYFX--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_yYFX_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_yYFX_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_yYFX_envp(%rip)
	nop
	movq	$23, -16(%rbp)
.L40:
	cmpq	$24, -16(%rbp)
	ja	.L44
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
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L44-.L8
	.long	.L45-.L8
	.long	.L44-.L8
	.long	.L20-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L19-.L8
	.long	.L44-.L8
	.long	.L18-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L17-.L8
	.long	.L44-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L44-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	movl	-28(%rbp), %eax
	cmpl	$20, %eax
	ja	.L24
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L26(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L26(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L26:
	.long	.L24-.L26
	.long	.L33-.L26
	.long	.L32-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L31-.L26
	.long	.L30-.L26
	.long	.L29-.L26
	.long	.L24-.L26
	.long	.L29-.L26
	.long	.L28-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L27-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L25-.L26
	.text
.L25:
	movq	$24, -16(%rbp)
	jmp	.L34
.L27:
	movq	$15, -16(%rbp)
	jmp	.L34
.L28:
	movq	$22, -16(%rbp)
	jmp	.L34
.L29:
	movq	$20, -16(%rbp)
	jmp	.L34
.L30:
	movq	$0, -16(%rbp)
	jmp	.L34
.L31:
	movq	$8, -16(%rbp)
	jmp	.L34
.L32:
	movq	$1, -16(%rbp)
	jmp	.L34
.L33:
	movq	$19, -16(%rbp)
	jmp	.L34
.L24:
	movq	$10, -16(%rbp)
	nop
.L34:
	jmp	.L35
.L16:
	movl	-24(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movsd	.LC0(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L19:
	movl	-24(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movsd	.LC2(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L22:
	movl	-24(%rbp), %edx
	movl	%edx, %eax
	sall	$3, %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	imulq	$-2004318071, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$5, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L9:
	movq	$13, -16(%rbp)
	jmp	.L35
.L15:
	movss	.LC3(%rip), %xmm0
	ucomiss	-20(%rbp), %xmm0
	jp	.L43
	movss	.LC3(%rip), %xmm0
	ucomiss	-20(%rbp), %xmm0
	je	.L37
.L43:
	movq	$17, -16(%rbp)
	jmp	.L35
.L37:
	movq	$5, -16(%rbp)
	jmp	.L35
.L7:
	movl	-24(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movsd	.LC4(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L17:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -16(%rbp)
	jmp	.L35
.L12:
	movl	-24(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movsd	.LC7(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L14:
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rcx
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movq	%rcx, %xmm0
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L35
.L10:
	movl	-24(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movsd	.LC9(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L20:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L35
.L18:
	movss	.LC3(%rip), %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L23:
	movl	-24(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movsd	.LC11(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L11:
	movl	-24(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movsd	.LC12(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L35
.L44:
	nop
.L35:
	jmp	.L40
.L45:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L42
	call	__stack_chk_fail@PLT
.L42:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC0:
	.long	-1889785610
	.long	1071183298
	.align 8
.LC1:
	.long	0
	.long	1078853632
	.align 8
.LC2:
	.long	-1030792151
	.long	1070344437
	.align 4
.LC3:
	.long	-1082130432
	.align 8
.LC4:
	.long	515396076
	.long	1070721925
	.align 8
.LC7:
	.long	171798692
	.long	1069589463
	.align 8
.LC9:
	.long	-1717986918
	.long	1070176665
	.align 8
.LC11:
	.long	-2061584302
	.long	1070092779
	.align 8
.LC12:
	.long	1546188227
	.long	1069925007
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
