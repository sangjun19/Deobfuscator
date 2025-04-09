	.file	"dhawansolanki_c_2_flatten.c"
	.text
	.globl	_TIG_IZ_hhYo_argv
	.bss
	.align 8
	.type	_TIG_IZ_hhYo_argv, @object
	.size	_TIG_IZ_hhYo_argv, 8
_TIG_IZ_hhYo_argv:
	.zero	8
	.globl	_TIG_IZ_hhYo_envp
	.align 8
	.type	_TIG_IZ_hhYo_envp, @object
	.size	_TIG_IZ_hhYo_envp, 8
_TIG_IZ_hhYo_envp:
	.zero	8
	.globl	_TIG_IZ_hhYo_argc
	.align 4
	.type	_TIG_IZ_hhYo_argc, @object
	.size	_TIG_IZ_hhYo_argc, 4
_TIG_IZ_hhYo_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the Coefficent of x^2 : "
.LC1:
	.string	"%f"
.LC2:
	.string	"Enter the Coefficent of x : "
.LC3:
	.string	"Enter the value of c : "
.LC5:
	.string	"Invalid Inputs."
	.align 8
.LC6:
	.string	"Roots are Distinct & Imaginary"
.LC11:
	.string	"Root 1 : %f + i%f \n"
.LC12:
	.string	"Root 2 : %f - i%f \n"
.LC14:
	.string	"Roots are Real & Distinct."
.LC15:
	.string	"Root 1 : %f \n"
.LC16:
	.string	"Root 2 : %f \n"
.LC17:
	.string	"Roots are Real & Equal."
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_hhYo_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_hhYo_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_hhYo_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 98 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-hhYo--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_hhYo_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_hhYo_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_hhYo_envp(%rip)
	nop
	movq	$0, -56(%rbp)
.L30:
	cmpq	$13, -56(%rbp)
	ja	.L38
	movq	-56(%rbp), %rax
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
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L39-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L38-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L7-.L8
	.text
.L14:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-76(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -56(%rbp)
	jmp	.L19
.L11:
	movss	-68(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	comiss	%xmm1, %xmm0
	jbe	.L36
	movq	$13, -56(%rbp)
	jmp	.L19
.L36:
	movq	$7, -56(%rbp)
	jmp	.L19
.L17:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -56(%rbp)
	jmp	.L19
.L15:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movss	-76(%rbp), %xmm0
	movss	.LC7(%rip), %xmm1
	xorps	%xmm1, %xmm0
	movss	.LC8(%rip), %xmm2
	movaps	%xmm0, %xmm1
	divss	%xmm2, %xmm1
	movss	-80(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -64(%rbp)
	movss	-68(%rbp), %xmm0
	movss	.LC9(%rip), %xmm1
	andps	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movsd	%xmm0, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movsd	-16(%rbp), %xmm0
	movsd	.LC10(%rip), %xmm2
	movapd	%xmm0, %xmm1
	divsd	%xmm2, %xmm1
	movss	-80(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	mulsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -60(%rbp)
	pxor	%xmm0, %xmm0
	cvtss2sd	-60(%rbp), %xmm0
	pxor	%xmm3, %xmm3
	cvtss2sd	-64(%rbp), %xmm3
	movq	%xmm3, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	pxor	%xmm0, %xmm0
	cvtss2sd	-60(%rbp), %xmm0
	pxor	%xmm4, %xmm4
	cvtss2sd	-64(%rbp), %xmm4
	movq	%xmm4, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	movq	$2, -56(%rbp)
	jmp	.L19
.L10:
	movss	-76(%rbp), %xmm1
	movss	-76(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	-80(%rbp), %xmm2
	movss	.LC13(%rip), %xmm1
	mulss	%xmm1, %xmm2
	movss	-72(%rbp), %xmm1
	mulss	%xmm2, %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -68(%rbp)
	movq	$8, -56(%rbp)
	jmp	.L19
.L7:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	pxor	%xmm5, %xmm5
	cvtss2sd	-68(%rbp), %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -48(%rbp)
	movss	-76(%rbp), %xmm0
	movss	.LC7(%rip), %xmm1
	xorps	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	addsd	-48(%rbp), %xmm1
	movss	-80(%rbp), %xmm0
	addss	%xmm0, %xmm0
	cvtss2sd	%xmm0, %xmm0
	divsd	%xmm0, %xmm1
	pxor	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	movss	%xmm0, -64(%rbp)
	pxor	%xmm6, %xmm6
	cvtss2sd	-68(%rbp), %xmm6
	movq	%xmm6, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -40(%rbp)
	movss	-76(%rbp), %xmm0
	movss	.LC7(%rip), %xmm1
	xorps	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	subsd	-40(%rbp), %xmm1
	movss	-80(%rbp), %xmm0
	addss	%xmm0, %xmm0
	cvtss2sd	%xmm0, %xmm0
	divsd	%xmm0, %xmm1
	pxor	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	movss	%xmm0, -60(%rbp)
	pxor	%xmm7, %xmm7
	cvtss2sd	-64(%rbp), %xmm7
	movq	%xmm7, %rax
	movq	%rax, %xmm0
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	pxor	%xmm3, %xmm3
	cvtss2sd	-60(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -56(%rbp)
	jmp	.L19
.L13:
	movss	-80(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jp	.L23
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jne	.L23
	movq	$1, -56(%rbp)
	jmp	.L19
.L23:
	movq	$9, -56(%rbp)
	jmp	.L19
.L9:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	pxor	%xmm4, %xmm4
	cvtss2sd	-68(%rbp), %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -32(%rbp)
	movss	-76(%rbp), %xmm0
	movss	.LC7(%rip), %xmm1
	xorps	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movapd	%xmm0, %xmm1
	addsd	-32(%rbp), %xmm1
	movss	-80(%rbp), %xmm0
	addss	%xmm0, %xmm0
	cvtss2sd	%xmm0, %xmm0
	divsd	%xmm0, %xmm1
	pxor	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	movss	%xmm0, -60(%rbp)
	movss	-60(%rbp), %xmm0
	movss	%xmm0, -64(%rbp)
	pxor	%xmm5, %xmm5
	cvtss2sd	-64(%rbp), %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	pxor	%xmm6, %xmm6
	cvtss2sd	-60(%rbp), %xmm6
	movq	%xmm6, %rax
	movq	%rax, %xmm0
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -56(%rbp)
	jmp	.L19
.L18:
	movq	$4, -56(%rbp)
	jmp	.L19
.L12:
	pxor	%xmm0, %xmm0
	comiss	-68(%rbp), %xmm0
	jbe	.L37
	movq	$3, -56(%rbp)
	jmp	.L19
.L37:
	movq	$10, -56(%rbp)
	jmp	.L19
.L38:
	nop
.L19:
	jmp	.L30
.L39:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L32
	call	__stack_chk_fail@PLT
.L32:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
	.align 16
.LC7:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.align 4
.LC8:
	.long	1073741824
	.align 16
.LC9:
	.long	2147483647
	.long	0
	.long	0
	.long	0
	.align 8
.LC10:
	.long	0
	.long	1073741824
	.align 4
.LC13:
	.long	1082130432
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
