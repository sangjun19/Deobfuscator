	.file	"gdicati_aulasFD2_cores_flatten.c"
	.text
	.globl	_TIG_IZ_d0U5_argv
	.bss
	.align 8
	.type	_TIG_IZ_d0U5_argv, @object
	.size	_TIG_IZ_d0U5_argv, 8
_TIG_IZ_d0U5_argv:
	.zero	8
	.globl	_TIG_IZ_d0U5_envp
	.align 8
	.type	_TIG_IZ_d0U5_envp, @object
	.size	_TIG_IZ_d0U5_envp, 8
_TIG_IZ_d0U5_envp:
	.zero	8
	.globl	_TIG_IZ_d0U5_argc
	.align 4
	.type	_TIG_IZ_d0U5_argc, @object
	.size	_TIG_IZ_d0U5_argc, 4
_TIG_IZ_d0U5_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%c[%d;%df"
	.text
	.globl	gotoxy
	.type	gotoxy, @function
gotoxy:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%edx, %ecx
	movl	%eax, %edx
	movl	$27, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	gotoxy, .-gotoxy
	.section	.rodata
.LC1:
	.string	"clear"
	.align 8
.LC2:
	.string	"\033[1;33m\033[33;5mTitle of the Program\033[0m"
	.align 8
.LC3:
	.string	"\033[1;31mHey this is the color red, and it's bold! \n\033[0m"
	.align 8
.LC4:
	.string	"\033[0;31mIf\033[0;34myou\033[0;33mare\033[0;32mbored\033[0;36mdo\033[0;35mthis! \n\033[0m"
	.align 8
.LC5:
	.string	"\033[4;36m\033[1;31mIf\033[1;34myou\033[1;33mare\033[1;32mbored\033[1;36mdo\033[1;35mthis! \n\033[0m"
	.align 8
.LC6:
	.string	"\033[4;31mIf\033[4;34myou\033[4;33mare\033[4;32mbored\033[4;36mdo\033[4;35mthis! \n\033[0m"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_d0U5_envp(%rip)
	nop
.L10:
	movq	$0, _TIG_IZ_d0U5_argv(%rip)
	nop
.L11:
	movl	$0, _TIG_IZ_d0U5_argc(%rip)
	nop
	nop
.L12:
.L13:
#APP
# 99 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-d0U5--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_d0U5_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_d0U5_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_d0U5_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L19:
	cmpq	$2, -8(%rbp)
	je	.L14
	cmpq	$2, -8(%rbp)
	ja	.L21
	cmpq	$0, -8(%rbp)
	je	.L16
	cmpq	$1, -8(%rbp)
	jne	.L21
	movq	$0, -8(%rbp)
	jmp	.L17
.L16:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	system@PLT
	movl	$10, %esi
	movl	$2, %edi
	call	gotoxy
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$8, %esi
	movl	$4, %edi
	call	gotoxy
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L17
.L14:
	movl	$0, %eax
	jmp	.L20
.L21:
	nop
.L17:
	jmp	.L19
.L20:
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
