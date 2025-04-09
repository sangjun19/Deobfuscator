	.file	"JoyChowdhury12_C-programme-basics_tutorial11_flatten.c"
	.text
	.globl	_TIG_IZ_ZrpA_argv
	.bss
	.align 8
	.type	_TIG_IZ_ZrpA_argv, @object
	.size	_TIG_IZ_ZrpA_argv, 8
_TIG_IZ_ZrpA_argv:
	.zero	8
	.globl	_TIG_IZ_ZrpA_envp
	.align 8
	.type	_TIG_IZ_ZrpA_envp, @object
	.size	_TIG_IZ_ZrpA_envp, 8
_TIG_IZ_ZrpA_envp:
	.zero	8
	.globl	_TIG_IZ_ZrpA_argc
	.align 4
	.type	_TIG_IZ_ZrpA_argc, @object
	.size	_TIG_IZ_ZrpA_argc, 4
_TIG_IZ_ZrpA_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Enter your age"
.LC1:
	.string	"%d"
.LC2:
	.string	"Enter your marks"
.LC3:
	.string	"The age is 23"
.LC4:
	.string	"The age is 3"
.LC5:
	.string	"your marks are not 45"
.LC6:
	.string	"Age is not 3, 13 or 23"
.LC7:
	.string	"The age is 13"
.LC8:
	.string	"Your marks are 45"
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
	movq	$0, _TIG_IZ_ZrpA_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ZrpA_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ZrpA_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ZrpA--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_ZrpA_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_ZrpA_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_ZrpA_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L28:
	cmpq	$13, -16(%rbp)
	ja	.L31
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
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L31-.L8
	.long	.L15-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	movl	-24(%rbp), %eax
	cmpl	$23, %eax
	je	.L19
	cmpl	$23, %eax
	jg	.L20
	cmpl	$3, %eax
	je	.L21
	cmpl	$13, %eax
	je	.L22
	jmp	.L20
.L19:
	movq	$11, -16(%rbp)
	jmp	.L23
.L22:
	movq	$7, -16(%rbp)
	jmp	.L23
.L21:
	movq	$13, -16(%rbp)
	jmp	.L23
.L20:
	movq	$0, -16(%rbp)
	nop
.L23:
	jmp	.L24
.L9:
	movl	-20(%rbp), %eax
	cmpl	$45, %eax
	jne	.L25
	movq	$2, -16(%rbp)
	jmp	.L26
.L25:
	movq	$10, -16(%rbp)
	nop
.L26:
	jmp	.L24
.L13:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$4, -16(%rbp)
	jmp	.L24
.L17:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	jmp	.L30
.L10:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L24
.L12:
	movq	$8, -16(%rbp)
	jmp	.L24
.L7:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L24
.L11:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L24
.L18:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L24
.L14:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L24
.L16:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L24
.L31:
	nop
.L24:
	jmp	.L28
.L30:
	call	__stack_chk_fail@PLT
.L29:
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
