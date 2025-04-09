	.file	"Abhranil2004_C-ALL-CODES-_1_flatten.c"
	.text
	.globl	_TIG_IZ_7Oe2_envp
	.bss
	.align 8
	.type	_TIG_IZ_7Oe2_envp, @object
	.size	_TIG_IZ_7Oe2_envp, 8
_TIG_IZ_7Oe2_envp:
	.zero	8
	.globl	_TIG_IZ_7Oe2_argc
	.align 4
	.type	_TIG_IZ_7Oe2_argc, @object
	.size	_TIG_IZ_7Oe2_argc, 4
_TIG_IZ_7Oe2_argc:
	.zero	4
	.globl	_TIG_IZ_7Oe2_argv
	.align 8
	.type	_TIG_IZ_7Oe2_argv, @object
	.size	_TIG_IZ_7Oe2_argv, 8
_TIG_IZ_7Oe2_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"It's Hot"
	.align 8
.LC1:
	.string	"Enter the temperature in celsius: "
.LC2:
	.string	"%f"
.LC5:
	.string	"Very cold weather"
.LC7:
	.string	"It's very Hot"
.LC9:
	.string	"Freezing weather"
.LC10:
	.string	"Cold weather"
.LC11:
	.string	"Normal in temp"
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
	movq	$0, _TIG_IZ_7Oe2_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_7Oe2_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_7Oe2_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 122 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-7Oe2--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_7Oe2_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_7Oe2_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_7Oe2_envp(%rip)
	nop
	movq	$10, -16(%rbp)
.L39:
	cmpq	$13, -16(%rbp)
	ja	.L52
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
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L17:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L22
.L9:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -16(%rbp)
	jmp	.L22
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L40
	jmp	.L46
.L20:
	movss	-20(%rbp), %xmm1
	movss	.LC3(%rip), %xmm0
	comiss	%xmm1, %xmm0
	jb	.L47
	movq	$4, -16(%rbp)
	jmp	.L22
.L47:
	movq	$13, -16(%rbp)
	jmp	.L22
.L18:
	movss	-20(%rbp), %xmm1
	movss	.LC4(%rip), %xmm0
	comiss	%xmm1, %xmm0
	jb	.L48
	movq	$7, -16(%rbp)
	jmp	.L22
.L48:
	movq	$1, -16(%rbp)
	jmp	.L22
.L10:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L22
.L12:
	movss	-20(%rbp), %xmm1
	movss	.LC6(%rip), %xmm0
	comiss	%xmm1, %xmm0
	jb	.L49
	movq	$0, -16(%rbp)
	jmp	.L22
.L49:
	movq	$3, -16(%rbp)
	jmp	.L22
.L7:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L22
.L15:
	movss	-20(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L50
	movq	$5, -16(%rbp)
	jmp	.L22
.L50:
	movq	$2, -16(%rbp)
	jmp	.L22
.L16:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L22
.L11:
	movq	$12, -16(%rbp)
	jmp	.L22
.L21:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L22
.L14:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -16(%rbp)
	jmp	.L22
.L19:
	movss	-20(%rbp), %xmm1
	movss	.LC12(%rip), %xmm0
	comiss	%xmm1, %xmm0
	jb	.L51
	movq	$11, -16(%rbp)
	jmp	.L22
.L51:
	movq	$9, -16(%rbp)
	jmp	.L22
.L52:
	nop
.L22:
	jmp	.L39
.L46:
	call	__stack_chk_fail@PLT
.L40:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC3:
	.long	1109393408
	.align 4
.LC4:
	.long	1106247680
	.align 4
.LC6:
	.long	1101004800
	.align 4
.LC12:
	.long	1092616192
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
