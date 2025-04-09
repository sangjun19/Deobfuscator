	.file	"Rajendran2201_C-Programming_Problem_07_flatten.c"
	.text
	.globl	_TIG_IZ_3tsz_envp
	.bss
	.align 8
	.type	_TIG_IZ_3tsz_envp, @object
	.size	_TIG_IZ_3tsz_envp, 8
_TIG_IZ_3tsz_envp:
	.zero	8
	.globl	_TIG_IZ_3tsz_argc
	.align 4
	.type	_TIG_IZ_3tsz_argc, @object
	.size	_TIG_IZ_3tsz_argc, 4
_TIG_IZ_3tsz_argc:
	.zero	4
	.globl	_TIG_IZ_3tsz_argv
	.align 8
	.type	_TIG_IZ_3tsz_argv, @object
	.size	_TIG_IZ_3tsz_argv, 8
_TIG_IZ_3tsz_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Sunday"
.LC1:
	.string	"Tuesday"
.LC2:
	.string	"Invalid input"
.LC3:
	.string	"Saturday"
.LC4:
	.string	"Thursday"
.LC5:
	.string	"Friday"
.LC6:
	.string	"Enter the number in integer: "
.LC7:
	.string	"%d"
.LC8:
	.string	"Monday"
.LC9:
	.string	"Wednesday"
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
	movq	$0, _TIG_IZ_3tsz_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_3tsz_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_3tsz_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-3tsz--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_3tsz_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_3tsz_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_3tsz_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L32:
	cmpq	$18, -16(%rbp)
	ja	.L35
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
	.long	.L19-.L8
	.long	.L35-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L35-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L35-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L35-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L35-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	-20(%rbp), %eax
	cmpl	$7, %eax
	ja	.L20
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L20-.L22
	.long	.L28-.L22
	.long	.L27-.L22
	.long	.L26-.L22
	.long	.L25-.L22
	.long	.L24-.L22
	.long	.L23-.L22
	.long	.L21-.L22
	.text
.L21:
	movq	$16, -16(%rbp)
	jmp	.L29
.L23:
	movq	$13, -16(%rbp)
	jmp	.L29
.L24:
	movq	$9, -16(%rbp)
	jmp	.L29
.L25:
	movq	$2, -16(%rbp)
	jmp	.L29
.L26:
	movq	$8, -16(%rbp)
	jmp	.L29
.L27:
	movq	$0, -16(%rbp)
	jmp	.L29
.L28:
	movq	$15, -16(%rbp)
	jmp	.L29
.L20:
	movq	$3, -16(%rbp)
	nop
.L29:
	jmp	.L30
.L16:
	movq	$10, -16(%rbp)
	jmp	.L30
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	jmp	.L34
.L15:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L17:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L14:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L11:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L13:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -16(%rbp)
	jmp	.L30
.L19:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L18:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L30
.L35:
	nop
.L30:
	jmp	.L32
.L34:
	call	__stack_chk_fail@PLT
.L33:
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
